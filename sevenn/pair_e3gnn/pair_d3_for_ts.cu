/*
Batched DFT-D3 dispersion correction for TorchSim.
Processes B systems in a single batch on one GPU.
Physics identical to pair_d3_for_ase.cu; only data layout differs.

Key differences from the single-system version:
  - All per-atom/per-pair arrays are flat (no pointer-of-pointer).
  - Tau arrays are flat and concatenated across systems.
  - Each kernel thread finds its system via binary search on offset arrays.
  - Sigma/disp use per-thread atomicAdd (no shared-memory block reduction)
    because threads within a block can belong to different systems.
*/

#include "pair_d3_for_ts.h"

/* --------- CUDA error handling macros --------- */
#define CHECK_CUDA(call) do {                                            \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA Error (%s:%d) -> %s: %s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
} while (0)

#define CHECK_CUDA_ERROR_STREAM(s) do {                                  \
    cudaError_t status_ = cudaStreamSynchronize(s);                      \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA Error (%s:%d) -> %s: %s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
} while (0)

/* --------- Math functions for CUDA compatibility --------- */

// Overflow-safe triangular unroll: linij -> (i, j) where i >= j
inline __host__ __device__ void ij_at_linij(int64_t linij, int &i, int &j) {
    double d = static_cast<double>(linij);
    i = static_cast<int>((sqrt(1.0 + 8.0 * d) - 1.0) / 2.0);
    // Guard against floating-point undershoot
    while (static_cast<int64_t>(i) * (i + 1) / 2 > linij) i--;
    while (static_cast<int64_t>(i + 1) * (i + 2) / 2 <= linij) i++;
    j = static_cast<int>(linij - static_cast<int64_t>(i) * (i + 1) / 2);
}

inline __host__ __device__ float lensq3(const float *v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
} // from MathExtra::lensq3

inline void cross3(const double *v1, const double *v2, double *ans)
{
  ans[0] = v1[1] * v2[2] - v1[2] * v2[1];
  ans[1] = v1[2] * v2[0] - v1[0] * v2[2];
  ans[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

inline double dot3(const double *v1, const double *v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

inline double len3(const double *v)
{
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
/* --------- Math functions for CUDA compatibility --------- */

// Binary search: find system index s such that offset[s] <= iter < offset[s+1]
inline __device__ int find_system(int64_t iter, const int64_t* offset, int B) {
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (offset[mid] <= iter) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// Same but for int offsets
inline __device__ int find_system_int(int iter, const int* offset, int B) {
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (offset[mid] <= iter) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

/* =========================================================================
   Constructor / Destructor
   ========================================================================= */

BatchPairD3::BatchPairD3() {
    cudaStreamCreate(&stream);
}

BatchPairD3::~BatchPairD3() {
    cudaStreamDestroy(stream);

    // D3 parameter arrays
    free_param_arrays();

    // Batch arrays (flat managed memory, single cudaFree each)
    if (atom_offset)     cudaFree(atom_offset);
    if (pair_offset)     cudaFree(pair_offset);
    if (tau_vdw_offset)  cudaFree(tau_vdw_offset);
    if (tau_cn_offset)   cudaFree(tau_cn_offset);
    if (center_tau_vdw)  cudaFree(center_tau_vdw);
    if (center_tau_cn)   cudaFree(center_tau_cn);

    if (d_atomtype)      cudaFree(d_atomtype);
    if (d_x)             cudaFree(d_x);
    if (d_cn)            cudaFree(d_cn);
    if (d_dc6i)          cudaFree(d_dc6i);
    if (d_f)             cudaFree(d_f);

    if (d_c6_ij_tot)     cudaFree(d_c6_ij_tot);
    if (d_dc6_iji_tot)   cudaFree(d_dc6_iji_tot);
    if (d_dc6_ijj_tot)   cudaFree(d_dc6_ijj_tot);

    if (d_disp)          cudaFree(d_disp);
    if (d_sigma)         cudaFree(d_sigma);

    if (d_tau_vdw)       cudaFree(d_tau_vdw);
    if (d_tau_cn)        cudaFree(d_tau_cn);
}

void BatchPairD3::free_param_arrays() {
    if (np1_save <= 0) return;
    int np1 = np1_save;
    cudaFree(r2r4); r2r4 = nullptr;
    cudaFree(rcov); rcov = nullptr;
    cudaFree(mxc);  mxc = nullptr;
    for (int i = 0; i < np1; i++) { cudaFree(r0ab[i]); }
    cudaFree(r0ab); r0ab = nullptr;
    for (int i = 0; i < np1; i++) {
        for (int j = 0; j < np1; j++) {
            for (int k = 0; k < MAXC; k++) {
                for (int l = 0; l < MAXC; l++) {
                    cudaFree(c6ab[i][j][k][l]);
                }
                cudaFree(c6ab[i][j][k]);
            }
            cudaFree(c6ab[i][j]);
        }
        cudaFree(c6ab[i]);
    }
    cudaFree(c6ab); c6ab = nullptr;
    np1_save = 0;
}

void BatchPairD3::alloc_param_arrays(int np1) {
    if (np1_save > 0) free_param_arrays();
    np1_save = np1;
    cudaMallocManaged(&r2r4, np1 * sizeof(float));
    cudaMallocManaged(&rcov, np1 * sizeof(float));
    cudaMallocManaged(&mxc, np1 * sizeof(int));
    cudaMallocManaged(&r0ab, np1 * sizeof(float*)); for (int i = 0; i < np1; i++) { cudaMallocManaged(&r0ab[i], np1 * sizeof(float)); }
    cudaMallocManaged(&c6ab, np1 * sizeof(float****));
    for (int i = 0; i < np1; i++) {
        cudaMallocManaged(&c6ab[i], np1 * sizeof(float***));
        for (int j = 0; j < np1; j++) {
            cudaMallocManaged(&c6ab[i][j], MAXC * sizeof(float**));
            for (int k = 0; k < MAXC; k++) {
                cudaMallocManaged(&c6ab[i][j][k], MAXC * sizeof(float*));
                for (int l = 0; l < MAXC; l++) {
                    cudaMallocManaged(&c6ab[i][j][k][l], 3 * sizeof(float));
                }
            }
        }
    }
    // Initialize c6ab to -1
    for (int i = 0; i < np1; i++)
        for (int j = 0; j < np1; j++)
            for (int k = 0; k < MAXC; k++)
                for (int l = 0; l < MAXC; l++)
                    for (int m = 0; m < 3; m++)
                        c6ab[i][j][k][l][m] = -1;
}

/* =========================================================================
   High-water-mark memory management for batch arrays
   ========================================================================= */

void BatchPairD3::ensure_capacity(int B, int N_total, int64_t P_total,
                                   int T_vdw_total, int T_cn_total) {
    // Offset arrays: need B+1
    if (B > alloc_B) {
        if (atom_offset)    cudaFree(atom_offset);
        if (pair_offset)    cudaFree(pair_offset);
        if (tau_vdw_offset) cudaFree(tau_vdw_offset);
        if (tau_cn_offset)  cudaFree(tau_cn_offset);
        if (center_tau_vdw) cudaFree(center_tau_vdw);
        if (center_tau_cn)  cudaFree(center_tau_cn);
        if (d_disp)         cudaFree(d_disp);
        if (d_sigma)        cudaFree(d_sigma);

        cudaMallocManaged(&atom_offset,    (B + 1) * sizeof(int));
        cudaMallocManaged(&pair_offset,    (B + 1) * sizeof(int64_t));
        cudaMallocManaged(&tau_vdw_offset, (B + 1) * sizeof(int));
        cudaMallocManaged(&tau_cn_offset,  (B + 1) * sizeof(int));
        cudaMallocManaged(&center_tau_vdw, B * sizeof(int));
        cudaMallocManaged(&center_tau_cn,  B * sizeof(int));
        cudaMallocManaged(&d_disp,         B * sizeof(double));
        cudaMallocManaged(&d_sigma,        B * 9 * sizeof(double));
        alloc_B = B;
    }

    if (N_total > alloc_N) {
        if (d_atomtype) cudaFree(d_atomtype);
        if (d_x)        cudaFree(d_x);
        if (d_cn)       cudaFree(d_cn);
        if (d_dc6i)     cudaFree(d_dc6i);
        if (d_f)        cudaFree(d_f);

        cudaMallocManaged(&d_atomtype, N_total * sizeof(int));
        cudaMallocManaged(&d_x,        N_total * 3 * sizeof(float));
        cudaMallocManaged(&d_cn,       N_total * sizeof(double));
        cudaMallocManaged(&d_dc6i,     N_total * sizeof(double));
        cudaMallocManaged(&d_f,        N_total * 3 * sizeof(double));
        alloc_N = N_total;
    }

    if (P_total > alloc_P) {
        if (d_c6_ij_tot)   cudaFree(d_c6_ij_tot);
        if (d_dc6_iji_tot) cudaFree(d_dc6_iji_tot);
        if (d_dc6_ijj_tot) cudaFree(d_dc6_ijj_tot);

        cudaMallocManaged(&d_c6_ij_tot,   P_total * sizeof(float));
        cudaMallocManaged(&d_dc6_iji_tot, P_total * sizeof(float));
        cudaMallocManaged(&d_dc6_ijj_tot, P_total * sizeof(float));
        alloc_P = P_total;
    }

    if (T_vdw_total > alloc_T_vdw) {
        if (d_tau_vdw) cudaFree(d_tau_vdw);
        cudaMallocManaged(&d_tau_vdw, T_vdw_total * 3 * sizeof(float));
        alloc_T_vdw = T_vdw_total;
    }

    if (T_cn_total > alloc_T_cn) {
        if (d_tau_cn) cudaFree(d_tau_cn);
        cudaMallocManaged(&d_tau_cn, T_cn_total * 3 * sizeof(float));
        alloc_T_cn = T_cn_total;
    }
}

/* =========================================================================
   Settings / Coeff (same parameter logic as original)
   ========================================================================= */

void BatchPairD3::settings(double vdw_sq, double cn_sq, std::string damp_name, std::string func_name) {
    rthr = vdw_sq;
    cnthr = cn_sq;

    std::map<std::string, int> commandMap = {
        {"damp_zero", 0}, {"damp_bj", 1}, {"damp_zerom", 2}, {"damp_bjm", 3},
    };

    auto it = commandMap.find(damp_name);
    if (it == commandMap.end()) {
        fprintf(stderr, "Error: Unknown damping function '%s'\n", damp_name.c_str());
        exit(EXIT_FAILURE);
    }
    damping = it->second;
    functional = func_name;
    setfuncpar();
}

/* ----------------------------------------------------------------------
   finds atomic number (used in BatchPairD3::coeff)
------------------------------------------------------------------------- */
int BatchPairD3::find_atomic_number(std::string& key) {
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    if (key.length() == 1) { key += " "; }
    key.resize(2);

    std::vector<std::string> element_table = {
        "h ","he",
        "li","be","b ","c ","n ","o ","f ","ne",
        "na","mg","al","si","p ","s ","cl","ar",
        "k ","ca","sc","ti","v ","cr","mn","fe","co","ni","cu",
        "zn","ga","ge","as","se","br","kr",
        "rb","sr","y ","zr","nb","mo","tc","ru","rh","pd","ag",
        "cd","in","sn","sb","te","i ","xe",
        "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb","dy",
        "ho","er","tm","yb","lu","hf","ta","w ","re","os","ir","pt",
        "au","hg","tl","pb","bi","po","at","rn",
        "fr","ra","ac","th","pa","u ","np","pu"
    };

    for (size_t i = 0; i < element_table.size(); ++i) {
        if (element_table[i] == key) {
            return static_cast<int>(i + 1);
        }
    }
    return -1;
}

/* ----------------------------------------------------------------------
   Check whether an integer value in an integer array (used in BatchPairD3::coeff)
------------------------------------------------------------------------- */

int BatchPairD3::is_int_in_array(int arr[], int size, int value) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == value) { return i; }
    }
    return -1;
}

/* ----------------------------------------------------------------------
   Read r0ab values from the table (used in BatchPairD3::coeff)
------------------------------------------------------------------------- */

void BatchPairD3::read_r0ab(int* atomic_numbers, int ntypes) {
    const double r0ab_table[94][94] = R0AB_TABLE;
    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            r0ab[i][j] = r0ab_table[atomic_numbers[i-1]-1][atomic_numbers[j-1]-1] / AU_TO_ANG;
        }
    }
}

/* ----------------------------------------------------------------------
   Get atom pair indices and grid indices (used in BatchPairD3::read_c6ab)
------------------------------------------------------------------------- */

void BatchPairD3::get_limit_in_pars_array(int& idx_atom_1, int& idx_atom_2, int& idx_i, int& idx_j) {
    const int shift = 100;

    idx_i = (idx_atom_1 - 1) / shift + 1;
    idx_j = (idx_atom_2 - 1) / shift + 1;

    idx_atom_1 = (idx_atom_1 - 1) % shift + 1;
    idx_atom_2 = (idx_atom_2 - 1) % shift + 1;

    // the code above replaces the code below
    //idx_i = 1;
    //idx_j = 1;
    //int shift = 100;
    //while (idx_atom_1 > shift) { idx_atom_1 -= shift; idx_i++; }
    //while (idx_atom_2 > shift) { idx_atom_2 -= shift; idx_j++; }
}

/* ----------------------------------------------------------------------
   Read c6ab values from the table (used in BatchPairD3::coeff)
------------------------------------------------------------------------- */

void BatchPairD3::read_c6ab(int* atomic_numbers, int ntypes) {
    for (int i = 1; i <= ntypes; i++) { mxc[i] = 0; }
    int grid_i = 0, grid_j = 0;

    const double c6ab_table[32385][5] = C6AB_TABLE;

    for (int i = 0; i < 32385; i++) {
        const double ref_c6 = c6ab_table[i][0];
        int atom_number_1 = static_cast<int>(c6ab_table[i][1]);
        int atom_number_2 = static_cast<int>(c6ab_table[i][2]);
        get_limit_in_pars_array(atom_number_1, atom_number_2, grid_i, grid_j);
        const int idx_atom_1 = is_int_in_array(atomic_numbers, ntypes, atom_number_1);
        if (idx_atom_1 < 0) { continue; }
        const int idx_atom_2 = is_int_in_array(atomic_numbers, ntypes, atom_number_2);
        if (idx_atom_2 < 0) { continue; }
        const double ref_cn1 = c6ab_table[i][3];
        const double ref_cn2 = c6ab_table[i][4];

        mxc[idx_atom_1 + 1] = std::max(mxc[idx_atom_1 + 1], grid_i);
        mxc[idx_atom_2 + 1] = std::max(mxc[idx_atom_2 + 1], grid_j);
        c6ab[idx_atom_1 + 1][idx_atom_2 + 1][grid_i - 1][grid_j - 1][0] = ref_c6;
        c6ab[idx_atom_1 + 1][idx_atom_2 + 1][grid_i - 1][grid_j - 1][1] = ref_cn1;
        c6ab[idx_atom_1 + 1][idx_atom_2 + 1][grid_i - 1][grid_j - 1][2] = ref_cn2;
        c6ab[idx_atom_2 + 1][idx_atom_1 + 1][grid_j - 1][grid_i - 1][0] = ref_c6;
        c6ab[idx_atom_2 + 1][idx_atom_1 + 1][grid_j - 1][grid_i - 1][1] = ref_cn2;
        c6ab[idx_atom_2 + 1][idx_atom_1 + 1][grid_j - 1][grid_i - 1][2] = ref_cn1;
    }
}

/* ----------------------------------------------------------------------
   Set functional parameters (used in BatchPairD3::coeff)
------------------------------------------------------------------------- */

void BatchPairD3::setfuncpar_zero() {
    s6 = 1.0;
    alp = 14.0;
    rs18 = 1.0;

    // default def2-QZVP (almost basis set limit)
    std::unordered_map<std::string, int> commandMap = {
    { "slater-dirac-exchange", 1}, { "b-lyp", 2 },    { "b-p", 3 },       { "b97-d", 4 },      { "revpbe", 5 },
    { "pbe", 6 },                  { "pbesol", 7 },   { "rpw86-pbe", 8 }, { "rpbe", 9 },       { "tpss", 10 },
    { "b3-lyp", 11 },              { "pbe0", 12 },    { "hse06", 13 },    { "revpbe38", 14 },  { "pw6b95", 15 },
    { "tpss0", 16 },               { "b2-plyp", 17 }, { "pwpb95", 18 },   { "b2gp-plyp", 19 }, { "ptpss", 20 },
    { "hf", 21 },                  { "mpwlyp", 22 },  { "bpbe", 23 },     { "bh-lyp", 24 },    { "tpssh", 25 },
    { "pwb6k", 26 },               { "b1b95", 27 },   { "bop", 28 },      { "o-lyp", 29 },     { "o-pbe", 30 },
    { "ssb", 31 },                 { "revssb", 32 },  { "otpss", 33 },    { "b3pw91", 34 },    { "revpbe0", 35 },
    { "pbe38", 36 },               { "mpw1b95", 37 }, { "mpwb1k", 38 },   { "bmk", 39 },       { "cam-b3lyp", 40 },
    { "lc-wpbe", 41 },             { "m05", 42 },     { "m052x", 43 },    { "m06l", 44 },      { "m06", 45 },
    { "m062x", 46 },               { "m06hf", 47 },   { "hcth120", 48 }
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
    case 1: rs6 = 0.999; s18 = -1.957; rs18 = 0.697; break;
    case 2: rs6 = 1.094; s18 = 1.682; break;
    case 3: rs6 = 1.139; s18 = 1.683; break;
    case 4: rs6 = 0.892; s18 = 0.909; break;
    case 5: rs6 = 0.923; s18 = 1.010; break;
    case 6: rs6 = 1.217; s18 = 0.722; break;
    case 7: rs6 = 1.345; s18 = 0.612; break;
    case 8: rs6 = 1.224; s18 = 0.901; break;
    case 9: rs6 = 0.872; s18 = 0.514; break;
    case 10: rs6 = 1.166; s18 = 1.105; break;
    case 11: rs6 = 1.261; s18 = 1.703; break;
    case 12: rs6 = 1.287; s18 = 0.928; break;
    case 13: rs6 = 1.129; s18 = 0.109; break;
    case 14: rs6 = 1.021; s18 = 0.862; break;
    case 15: rs6 = 1.532; s18 = 0.862; break;
    case 16: rs6 = 1.252; s18 = 1.242; break;
    case 17: rs6 = 1.427; s18 = 1.022; s6 = 0.64; break;
    case 18: rs6 = 1.557; s18 = 0.705; s6 = 0.82; break;
    case 19: rs6 = 1.586; s18 = 0.760; s6 = 0.56; break;
    case 20: rs6 = 1.541; s18 = 0.879; s6 = 0.75; break;
    case 21: rs6 = 1.158; s18 = 1.746; break;
    case 22: rs6 = 1.239; s18 = 1.098; break;
    case 23: rs6 = 1.087; s18 = 2.033; break;
    case 24: rs6 = 1.370; s18 = 1.442; break;
    case 25: rs6 = 1.223; s18 = 1.219; break;
    case 26: rs6 = 1.660; s18 = 0.550; break;
    case 27: rs6 = 1.613; s18 = 1.868; break;
    case 28: rs6 = 0.929; s18 = 1.975; break;
    case 29: rs6 = 0.806; s18 = 1.764; break;
    case 30: rs6 = 0.837; s18 = 2.055; break;
    case 31: rs6 = 1.215; s18 = 0.663; break;
    case 32: rs6 = 1.221; s18 = 0.560; break;
    case 33: rs6 = 1.128; s18 = 1.494; break;
    case 34: rs6 = 1.176; s18 = 1.775; break;
    case 35: rs6 = 0.949; s18 = 0.792; break;
    case 36: rs6 = 1.333; s18 = 0.998; break;
    case 37: rs6 = 1.605; s18 = 1.118; break;
    case 38: rs6 = 1.671; s18 = 1.061; break;
    case 39: rs6 = 1.931; s18 = 2.168; break;
    case 40: rs6 = 1.378; s18 = 1.217; break;
    case 41: rs6 = 1.355; s18 = 1.279; break;
    case 42: rs6 = 1.373; s18 = 0.595; break;
    case 43: rs6 = 1.417; s18 = 0.000; break;
    case 44: rs6 = 1.581; s18 = 0.000; break;
    case 45: rs6 = 1.325; s18 = 0.000; break;
    case 46: rs6 = 1.619; s18 = 0.000; break;
    case 47: rs6 = 1.446; s18 = 0.000; break;
    /* DFTB3(zeta = 4.0), old deprecated parameters; case ("dftb3"); rs6 = 1.235; s18 = 0.673; */
    case 48: rs6 = 1.221; s18 = 1.206; break;
    default:
        fprintf(stderr, "Error: Functional name unknown for zero damping\n");
        exit(EXIT_FAILURE);
    }
}

void BatchPairD3::setfuncpar_bj() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b-p", 1}, {"b-lyp", 2}, {"revpbe", 3}, {"rpbe", 4}, {"b97-d", 5}, {"pbe", 6},
        {"rpw86-pbe", 7}, {"b3-lyp", 8}, {"tpss", 9}, {"hf", 10}, {"tpss0", 11}, {"pbe0", 12},
        {"hse06", 13}, {"revpbe38", 14}, {"pw6b95", 15}, {"b2-plyp", 16}, {"dsd-blyp", 17},
        {"dsd-blyp-fc", 18}, {"bop", 19}, {"mpwlyp", 20}, {"o-lyp", 21}, {"pbesol", 22}, {"bpbe", 23},
        {"opbe", 24}, {"ssb", 25}, {"revssb", 26}, {"otpss", 27}, {"b3pw91", 28}, {"bh-lyp", 29},
        {"revpbe0", 30}, {"tpssh", 31}, {"mpw1b95", 32}, {"pwb6k", 33}, {"b1b95", 34}, {"bmk", 35},
        {"cam-b3lyp", 36}, {"lc-wpbe", 37}, {"b2gp-plyp", 38}, {"ptpss", 39}, {"pwpb95", 40},
        {"hf/mixed", 41}, {"hf/sv", 42}, {"hf/minis", 43}, {"b3-lyp/6-31gd", 44}, {"hcth120", 45},
        {"pw1pw", 46}, {"pwgga", 47}, {"hsesol", 48}, {"hf3c", 49}, {"hf3cv", 50}, {"pbeh3c", 51},
        {"pbeh-3c", 52}, {"wb97m", 53}, {"r2scan", 54}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 0.3946; s18 = 3.2822; rs18 = 4.8516; break;
        case 2: rs6 = 0.4298; s18 = 2.6996; rs18 = 4.2359; break;
        case 3: rs6 = 0.5238; s18 = 2.3550; rs18 = 3.5016; break;
        case 4: rs6 = 0.1820; s18 = 0.8318; rs18 = 4.0094; break;
        case 5: rs6 = 0.5545; s18 = 2.2609; rs18 = 3.2297; break;
        case 6: rs6 = 0.4289; s18 = 0.7875; rs18 = 4.4407; break;
        case 7: rs6 = 0.4613; s18 = 1.3845; rs18 = 4.5062; break;
        case 8: rs6 = 0.3981; s18 = 1.9889; rs18 = 4.4211; break;
        case 9: rs6 = 0.4535; s18 = 1.9435; rs18 = 4.4752; break;
        case 10: rs6 = 0.3385; s18 = 0.9171; rs18 = 2.8830; break;
        case 11: rs6 = 0.3768; s18 = 1.2576; rs18 = 4.5865; break;
        case 12: rs6 = 0.4145; s18 = 1.2177; rs18 = 4.8593; break;
        case 13: rs6 = 0.383; s18 = 2.310; rs18 = 5.685; break;
        case 14: rs6 = 0.4309; s18 = 1.4760; rs18 = 3.9446; break;
        case 15: rs6 = 0.2076; s18 = 0.7257; rs18 = 6.3750; break;
        case 16: rs6 = 0.3065; s18 = 0.9147; rs18 = 5.0570; break; s6 = 0.64;
        case 17: rs6 = 0.0000; s18 = 0.2130; rs18 = 6.0519; s6 = 0.50; break;
        case 18: rs6 = 0.0009; s18 = 0.2112; rs18 = 5.9807; s6 = 0.50; break;
        case 19: rs6 = 0.4870; s18 = 3.2950; rs18 = 3.5043; break;
        case 20: rs6 = 0.4831; s18 = 2.0077; rs18 = 4.5323; break;
        case 21: rs6 = 0.5299; s18 = 2.6205; rs18 = 2.8065; break;
        case 22: rs6 = 0.4466; s18 = 2.9491; rs18 = 6.1742; break;
        case 23: rs6 = 0.4567; s18 = 4.0728; rs18 = 4.3908; break;
        case 24: rs6 = 0.5512; s18 = 3.3816; rs18 = 2.9444; break;
        case 25: rs6 = -0.0952; s18 = -0.1744; rs18 = 5.2170; break;
        case 26: rs6 = 0.4720; s18 = 0.4389; rs18 = 4.0986; break;
        case 27: rs6 = 0.4634; s18 = 2.7495; rs18 = 4.3153; break;
        case 28: rs6 = 0.4312; s18 = 2.8524; rs18 = 4.4693; break;
        case 29: rs6 = 0.2793; s18 = 1.0354; rs18 = 4.9615; break;
        case 30: rs6 = 0.4679; s18 = 1.7588; rs18 = 3.7619; break;
        case 31: rs6 = 0.4529; s18 = 2.2382; rs18 = 4.6550; break;
        case 32: rs6 = 0.1955; s18 = 1.0508; rs18 = 6.4177; break;
        case 33: rs6 = 0.1805; s18 = 0.9383; rs18 = 7.7627; break;
        case 34: rs6 = 0.2092; s18 = 1.4507; rs18 = 5.5545; break;
        case 35: rs6 = 0.1940; s18 = 2.0860; rs18 = 5.9197; break;
        case 36: rs6 = 0.3708; s18 = 2.0674; rs18 = 5.4743; break;
        case 37: rs6 = 0.3919; s18 = 1.8541; rs18 = 5.0897; break;
        case 38: rs6 = 0.0000; s18 = 0.2597; rs18 = 6.3332; s6 = 0.560; break;
        case 39: rs6 = 0.0000; s18 = 0.2804; rs18 = 6.5745; s6 = 0.750; break;
        case 40: rs6 = 0.0000; s18 = 0.2904; rs18 = 7.3141; s6 = 0.820; break;
        // special HF / DFT with eBSSE correction;
        case 41: rs6 = 0.5607; s18 = 3.9027; rs18 = 4.5622; break;
        case 42: rs6 = 0.4249; s18 = 2.1849; rs18 = 4.2783; break;
        case 43: rs6 = 0.1702; s18 = 0.9841; rs18 = 3.8506; break;
        case 44: rs6 = 0.5014; s18 = 4.0672; rs18 = 4.8409; break;
        case 45: rs6 = 0.3563; s18 = 1.0821; rs18 = 4.3359; break;
        /*     DFTB3 old, deprecated parameters : ;
            *     case ("dftb3"); rs6 = 0.7461; s18 = 3.209; rs18 = 4.1906;
            *     special SCC - DFTB parametrization;
            *     full third order DFTB, self consistent charges, hydrogen pair damping with; exponent 4.2;
        */
        case 46: rs6 = 0.3807; s18 = 2.3363; rs18 = 5.8844; break;
        case 47: rs6 = 0.2211; s18 = 2.6910; rs18 = 6.7278; break;
        case 48: rs6 = 0.4650; s18 = 2.9215; rs18 = 6.2003; break;
        // special HF - D3 - gCP - SRB / MINIX parametrization;
        case 49: rs6 = 0.4171; s18 = 0.8777; rs18 = 2.9149; break;
        // special HF - D3 - gCP - SRB2 / ECP - 2G parametrization;
        case 50: rs6 = 0.3063; s18 = 0.5022; rs18 = 3.9856; break;
        // special PBEh - D3 - gCP / def2 - mSVP parametrization;
        case 51: rs6 = 0.4860; s18 = 0.0000; rs18 = 4.5000; break;
        case 52: rs6 = 0.4860; s18 = 0.0000; rs18 = 4.5000; break;
        case 53: rs6 = 0.5660; s18 = 0.3908; rs18 = 3.1280; break;  // J. Chem. Theory Comput. 14, 5725 (2018)
        case 54: rs6 = 0.4948; s18 = 0.7898; rs18 = 5.7308; break;  // S. Ehlert et al.,  J. Chem. Phys. 154, 061101 (2021)
        default:
            fprintf(stderr, "Error: Functional name unknown for BJ damping\n");
            exit(EXIT_FAILURE);
    }
}

void BatchPairD3::setfuncpar_zerom() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
        {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 1.313134; s18 = 0.717543; rs18 = 0.016035; s6 = 0.640000; break;
        case 2: rs6 = 1.338153; s18 = 1.532981; rs18 = 0.013988; break;
        case 3: rs6 = 1.151808; s18 = 1.020078; rs18 = 0.035964; break;
        case 4: rs6 = 1.279637; s18 = 1.841686; rs18 = 0.014370; break;
        case 5: rs6 = 1.233460; s18 = 1.945174; rs18 = 0.000000; break;
        case 6: rs6 = 2.340218; s18 = 0.000000; rs18 = 0.129434; break;
        case 7: rs6 = 2.077949; s18 = 0.000081; rs18 = 0.116755; break;
        case 8: rs6 = 1.366361; s18 = 1.280619; rs18 = 0.003160; break;
        default:
            fprintf(stderr, "Error: Functional name unknown for zerom damping\n");
            exit(EXIT_FAILURE);
    }
}

void BatchPairD3::setfuncpar_bjm() {
    s6 = 1.0;
    alp = 14.0;

    std::unordered_map<std::string, int> commandMap = {
        {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
        {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
    };

    int commandCode = commandMap[functional];
    switch (commandCode) {
        case 1: rs6 = 0.486434; s18 = 0.672820; rs18 = 3.656466; s6 = 0.640000; break;
        case 2: rs6 = 0.278672; s18 = 1.466677; rs18 = 4.606311; break;
        case 3: rs6 = 0.240184; s18 = 1.206988; rs18 = 3.864426; break;
        case 4: rs6 = 0.448486; s18 = 1.875007; rs18 = 3.610679; break;
        case 5: rs6 = 0.821850; s18 = 3.140281; rs18 = 2.728151; break;
        case 6: rs6 = 0.012092; s18 = 0.358940; rs18 = 5.938951; break;
        case 7: rs6 = 0.007912; s18 = 0.528823; rs18 = 6.162326; break;
        case 8: rs6 = 0.563761; s18 = 0.906564; rs18 = 3.593680; break;
        default:
            fprintf(stderr, "Error: Functional name unknown for bjm damping\n");
            exit(EXIT_FAILURE);
    }
}

void BatchPairD3::setfuncpar() {
    void (BatchPairD3::*setfuncpar_damp[4])() = {
        &BatchPairD3::setfuncpar_zero,
        &BatchPairD3::setfuncpar_bj,
        &BatchPairD3::setfuncpar_zerom,
        &BatchPairD3::setfuncpar_bjm
    };
    (this->*setfuncpar_damp[damping])();

    rs8 = rs18;
    alp6 = alp;
    alp8 = alp + 2.0;
    // rs10 = rs18
    // alp10 = alp + 4.0;

    a1 = rs6;
    a2 = rs8;
    s8 = s18;
    // s6 is already defined
}

/* ----------------------------------------------------------------------
   Coeff : read from pair_coeff (Required) -> pair_coeff * * element1 element2 ...
------------------------------------------------------------------------- */

void BatchPairD3::coeff(int* atomic_numbers, int ntypes) {
    int np1 = ntypes + 1;
    if (np1 != np1_save) {
        alloc_param_arrays(np1);
    }
    /*
    scale r4/r2 values of the atoms by sqrt(Z)
    sqrt is also globally close to optimum
    together with the factor 1/2 this yield reasonable
    c8 for he, ne and ar. for larger Z, C8 becomes too large
    which effectively mimics higher R^n terms neglected due
    to stability reasons

    r2r4 =sqrt(0.5*r2r4(i)*dfloat(i)**0.5 ) with i=elementnumber
    the large number of digits is just to keep the results consistent
    with older versions. They should not imply any higher accuracy than
    the old values
    */
    double r2r4_ref[94] = {
         2.00734898,  1.56637132,  5.01986934,  3.85379032,  3.64446594,
         3.10492822,  2.71175247,  2.59361680,  2.38825250,  2.21522516,
         6.58585536,  5.46295967,  5.65216669,  4.88284902,  4.29727576,
         4.04108902,  3.72932356,  3.44677275,  7.97762753,  7.07623947,
         6.60844053,  6.28791364,  6.07728703,  5.54643096,  5.80491167,
         5.58415602,  5.41374528,  5.28497229,  5.22592821,  5.09817141,
         6.12149689,  5.54083734,  5.06696878,  4.87005108,  4.59089647,
         4.31176304,  9.55461698,  8.67396077,  7.97210197,  7.43439917,
         6.58711862,  6.19536215,  6.01517290,  5.81623410,  5.65710424,
         5.52640661,  5.44263305,  5.58285373,  7.02081898,  6.46815523,
         5.98089120,  5.81686657,  5.53321815,  5.25477007, 11.02204549,
        10.15679528,  9.35167836,  9.06926079,  8.97241155,  8.90092807,
         8.85984840,  8.81736827,  8.79317710,  7.89969626,  8.80588454,
         8.42439218,  8.54289262,  8.47583370,  8.45090888,  8.47339339,
         7.83525634,  8.20702843,  7.70559063,  7.32755997,  7.03887381,
         6.68978720,  6.05450052,  5.88752022,  5.70661499,  5.78450695,
         7.79780729,  7.26443867,  6.78151984,  6.67883169,  6.39024318,
         6.09527958, 11.79156076, 11.10997644,  9.51377795,  8.67197068,
         8.77140725,  8.65402716,  8.53923501,  8.85024712
    };

    /*
    covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197)
    values for metals decreased by 10 %
    !      data rcov/
    !     .  0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67
    !     ., 1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54
    !     ., 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09
    !     ., 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39
    !     ., 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26
    !     ., 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58, 1.57
    !     ., 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53
    !     ., 1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32
    !     ., 1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58
    !     ., 1.52, 1.53, 1.54, 1.55 /

    these new data are scaled with k2=4./3.  and converted a_0 via
    autoang=0.52917726d0
    */

    double rcov_ref[94] = {
        0.80628308, 1.15903197, 3.02356173, 2.36845659, 1.94011865,
        1.88972601, 1.78894056, 1.58736983, 1.61256616, 1.68815527,
        3.52748848, 3.14954334, 2.84718717, 2.62041997, 2.77159820,
        2.57002732, 2.49443835, 2.41884923, 4.43455700, 3.88023730,
        3.35111422, 3.07395437, 3.04875805, 2.77159820, 2.69600923,
        2.62041997, 2.51963467, 2.49443835, 2.54483100, 2.74640188,
        2.82199085, 2.74640188, 2.89757982, 2.77159820, 2.87238349,
        2.94797246, 4.76210950, 4.20778980, 3.70386304, 3.50229216,
        3.32591790, 3.12434702, 2.89757982, 2.84718717, 2.84718717,
        2.72120556, 2.89757982, 3.09915070, 3.22513231, 3.17473967,
        3.17473967, 3.09915070, 3.32591790, 3.30072128, 5.26603625,
        4.43455700, 4.08180818, 3.70386304, 3.98102289, 3.95582657,
        3.93062995, 3.90543362, 3.80464833, 3.82984466, 3.80464833,
        3.77945201, 3.75425569, 3.75425569, 3.72905937, 3.85504098,
        3.67866672, 3.45189952, 3.30072128, 3.09915070, 2.97316878,
        2.92277614, 2.79679452, 2.82199085, 2.84718717, 3.32591790,
        3.27552496, 3.27552496, 3.42670319, 3.30072128, 3.47709584,
        3.57788113, 5.06446567, 4.56053862, 4.20778980, 3.98102289,
        3.82984466, 3.85504098, 3.88023730, 3.90543362
    }; // covalent radii

    for (int i = 0; i < ntypes; i++) {
        r2r4[i+1] = r2r4_ref[atomic_numbers[i]-1];
        rcov[i+1] = rcov_ref[atomic_numbers[i]-1];
    }

    // set r0ab
    read_r0ab(atomic_numbers, ntypes);

    // read c6ab
    read_c6ab(atomic_numbers, ntypes);
    params_set = true;
}

/* =========================================================================
   Lattice helpers (CPU-side, per system)
   ========================================================================= */

void BatchPairD3::compute_lattice_reps(const double lat_v[3][3],
                                        const int pbc_flags[3],
                                        float r_threshold, int rep_out[3]) {
    double r_cutoff = sqrt(r_threshold);
    // lat_v is column-major: lat_v[col][row], i.e. lat_v_1 = lat_v[0]
    double lat_cp_12[3], lat_cp_23[3], lat_cp_31[3];
    cross3(lat_v[0], lat_v[1], lat_cp_12);
    cross3(lat_v[1], lat_v[2], lat_cp_23);
    cross3(lat_v[2], lat_v[0], lat_cp_31);

    double cos_value;
    cos_value = dot3(lat_cp_23, lat_v[0]) / len3(lat_cp_23);
    rep_out[0] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
    cos_value = dot3(lat_cp_31, lat_v[1]) / len3(lat_cp_31);
    rep_out[1] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
    cos_value = dot3(lat_cp_12, lat_v[2]) / len3(lat_cp_12);
    rep_out[2] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;

    if (pbc_flags[0] == 0) rep_out[0] = 0;
    if (pbc_flags[1] == 0) rep_out[1] = 0;
    if (pbc_flags[2] == 0) rep_out[2] = 0;
}

void BatchPairD3::compute_tau_flat(const double lat_v[3][3], const int rep[3],
                                    float* tau_buf, int n_tau, int& center_idx) {
    int xlim = rep[0], ylim = rep[1], zlim = rep[2];
    int idx = 0;
    center_idx = -1;
    for (int taux = -xlim; taux <= xlim; taux++) {
        for (int tauy = -ylim; tauy <= ylim; tauy++) {
            for (int tauz = -zlim; tauz <= zlim; tauz++) {
                if (taux == 0 && tauy == 0 && tauz == 0) {
                    center_idx = idx;
                }
                tau_buf[idx * 3 + 0] = lat_v[0][0] * taux + lat_v[1][0] * tauy + lat_v[2][0] * tauz;
                tau_buf[idx * 3 + 1] = lat_v[0][1] * taux + lat_v[1][1] * tauy + lat_v[2][1] * tauz;
                tau_buf[idx * 3 + 2] = lat_v[0][2] * taux + lat_v[1][2] * tauy + lat_v[2][2] * tauz;
                idx++;
            }
        }
    }
    // If all reps are 0, center_idx = 0 (only one tau = (0,0,0))
    if (center_idx < 0) center_idx = 0;
}

/* =========================================================================
   CUDA Kernels
   ========================================================================= */

/* ---- Kernel 1: Coordination Number ---- */
__global__ void batch_kernel_get_coordination_number(
    int64_t total_pairs, int B,
    float cnthr, float K1,
    float *rcov, int *atomtype,
    float *x, double *cn,
    int *atom_offset, int64_t *pair_offset,
    float *tau_cn, int *tau_cn_offset, int *center_tau_cn
) {
    int64_t iter = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (iter >= total_pairs) return;

    // Find which system this pair belongs to
    int sys = find_system(iter, pair_offset, B);
    int64_t local_iter = iter - pair_offset[sys];
    int n_sys = atom_offset[sys + 1] - atom_offset[sys];
    int64_t maxij_sys = static_cast<int64_t>(n_sys) * (n_sys + 1) / 2;
    if (local_iter >= maxij_sys) return;

    int iat_local, jat_local;
    ij_at_linij(local_iter, iat_local, jat_local);

    int iat = iat_local + atom_offset[sys];
    int jat = jat_local + atom_offset[sys];

    int tau_start = tau_cn_offset[sys];
    int n_tau = tau_cn_offset[sys + 1] - tau_start;
    int center_t = center_tau_cn[sys];

    float cn_local = 0.0f;

    if (iat == jat) {
        const float rcov_sum = rcov[atomtype[iat]] * 2.0f;
        for (int t = 0; t < n_tau; t++) {
            if (t == center_t) continue;
            const float rx = tau_cn[(tau_start + t) * 3 + 0];
            const float ry = tau_cn[(tau_start + t) * 3 + 1];
            const float rz = tau_cn[(tau_start + t) * 3 + 2];
            const float r2 = rx * rx + ry * ry + rz * rz;
            if (r2 <= cnthr) {
                const float r_rc = rsqrtf(r2);
                const float damp = 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
                cn_local += damp;
            }
        }
        atomicAdd(&cn[iat], cn_local);
    } else {
        const float rcov_sum = rcov[atomtype[iat]] + rcov[atomtype[jat]];
        for (int t = 0; t < n_tau; t++) {
            const float rx = x[jat * 3 + 0] - x[iat * 3 + 0] + tau_cn[(tau_start + t) * 3 + 0];
            const float ry = x[jat * 3 + 1] - x[iat * 3 + 1] + tau_cn[(tau_start + t) * 3 + 1];
            const float rz = x[jat * 3 + 2] - x[iat * 3 + 2] + tau_cn[(tau_start + t) * 3 + 2];
            const float r2 = rx * rx + ry * ry + rz * rz;
            if (r2 <= cnthr) {
                const float r_rc = rsqrtf(r2);
                const float damp = 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
                cn_local += damp;
            }
        }
        atomicAdd(&cn[iat], cn_local);
        atomicAdd(&cn[jat], cn_local);
    }
}

/* ---- Kernel 2: dC6/dCN ---- */
__global__ void batch_kernel_get_dC6_dCNij(
    int64_t total_pairs, int B, float K3,
    double *cn, int *mxc, float *****c6ab, int *atomtype,
    float *c6_ij_tot, float *dc6_iji_tot, float *dc6_ijj_tot,
    int *atom_offset, int64_t *pair_offset
) {
    int64_t iter = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (iter >= total_pairs) return;

    int sys = find_system(iter, pair_offset, B);
    int64_t local_iter = iter - pair_offset[sys];
    int n_sys = atom_offset[sys + 1] - atom_offset[sys];
    int64_t maxij_sys = static_cast<int64_t>(n_sys) * (n_sys + 1) / 2;
    if (local_iter >= maxij_sys) return;

    int iat_local, jat_local;
    ij_at_linij(local_iter, iat_local, jat_local);

    int iat = iat_local + atom_offset[sys];
    int jat = jat_local + atom_offset[sys];

    const int atomtype_i = atomtype[iat];
    const int atomtype_j = atomtype[jat];

    const float cni = cn[iat];
    const int mxci = mxc[atomtype_i];
    const float cnj = cn[jat];
    const int mxcj = mxc[atomtype_j];

    float c6mem = -1e99f;
    float r_save = 9999.0f;
    double numerator = 0.0;
    double denominator = 0.0;
    double d_numerator_i = 0.0;
    double d_denominator_i = 0.0;
    double d_numerator_j = 0.0;
    double d_denominator_j = 0.0;

    for (int a = 0; a < mxci; a++) {
        for (int b = 0; b < mxcj; b++) {
            float c6ref = c6ab[atomtype_i][atomtype_j][a][b][0];
            if (c6ref > 0.0f) {
                float cn_refi = c6ab[atomtype_i][atomtype_j][a][b][1];
                float cn_refj = c6ab[atomtype_i][atomtype_j][a][b][2];

                float r = (cn_refi - cni) * (cn_refi - cni) + (cn_refj - cnj) * (cn_refj - cnj);
                if (r < r_save) {
                    r_save = r;
                    c6mem = c6ref;
                }

                double expterm = exp(static_cast<double>(K3) * static_cast<double>(r));
                numerator += c6ref * expterm;
                denominator += expterm;

                expterm *= 2.0f * K3;

                double term = expterm * (cni - cn_refi);
                d_numerator_i += c6ref * term;
                d_denominator_i += term;

                term = expterm * (cnj - cn_refj);
                d_numerator_j += c6ref * term;
                d_denominator_j += term;
            }
        }
    }

    if (denominator > 1e-99) {
        const double denominator_rc = 1.0 / denominator;
        const double unit_frac = numerator * denominator_rc;
        c6_ij_tot[iter] = unit_frac;
        dc6_iji_tot[iter] = denominator_rc * fma(unit_frac, -d_denominator_i, d_numerator_i);
        dc6_ijj_tot[iter] = denominator_rc * fma(unit_frac, -d_denominator_j, d_numerator_j);
    } else {
        c6_ij_tot[iter] = c6mem;
        dc6_iji_tot[iter] = 0.0f;
        dc6_ijj_tot[iter] = 0.0f;
    }
}

/* ---- Kernel 3a: Forces without dC6 (zero damping) ---- */
__global__ void batch_kernel_forces_zero(
    int64_t total_pairs, int B,
    float rthr, float s6, float s8, float a1, float a2, float alp6, float alp8,
    float *r2r4, float **r0ab, int *atomtype,
    float *x, float *c6_ij_tot, float *dc6_iji_tot, float *dc6_ijj_tot,
    double *dc6i, double *disp, double *f, double *sigma,
    int *atom_offset, int64_t *pair_offset,
    float *tau_vdw, int *tau_vdw_offset, int *center_tau_vdw
) {
    int64_t iter = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (iter >= total_pairs) return;

    int sys = find_system(iter, pair_offset, B);
    int64_t local_iter = iter - pair_offset[sys];
    int n_sys = atom_offset[sys + 1] - atom_offset[sys];
    int64_t maxij_sys = static_cast<int64_t>(n_sys) * (n_sys + 1) / 2;
    if (local_iter >= maxij_sys) return;

    int iat_local, jat_local;
    ij_at_linij(local_iter, iat_local, jat_local);

    int iat = iat_local + atom_offset[sys];
    int jat = jat_local + atom_offset[sys];

    int tau_start = tau_vdw_offset[sys];
    int n_tau = tau_vdw_offset[sys + 1] - tau_start;
    int center_t = center_tau_vdw[sys];

    float f_local[3] = { 0.0f };
    float dc6i_local_i = 0.0f;
    float dc6i_local_j = 0.0f;
    float sigma_local[9] = { 0.0f };
    float disp_local = 0.0f;

    const float c6 = c6_ij_tot[iter];
    const float dc6iji = dc6_iji_tot[iter];
    const float dc6ijj = dc6_ijj_tot[iter];

    if (iat == jat) {
        const int atomtype_i = atomtype[iat];
        const float r0 = r0ab[atomtype_i][atomtype_i];
        const float unit_r2r4 = r2r4[atomtype_i];
        const float r42 = unit_r2r4 * unit_r2r4;
        const float unit_a1 = (a1 * r0);
        const float unit_a2 = (a2 * r0);
        const float s8r42 = s8 * r42;

        for (int t = 0; t < n_tau; t++) {
            if (t == center_t) continue;
            const float rij[3] = {
                tau_vdw[(tau_start + t) * 3 + 0],
                tau_vdw[(tau_start + t) * 3 + 1],
                tau_vdw[(tau_start + t) * 3 + 2]
            };
            const float r2 = lensq3(rij);
            if (r2 > rthr) continue;

            const float r_rc = rsqrtf(r2);
            float unit_rc_a1 = unit_a1 * r_rc;
            float t6 = unit_rc_a1 * unit_rc_a1;
            t6 *= unit_rc_a1;
            t6 *= t6;
            t6 *= unit_rc_a1;
            t6 *= t6;
            const float damp6 = 1.0f / fmaf(t6, 6.0f, 1.0f);
            float unit_rc_a2 = unit_a2 * r_rc;
            float t8 = unit_rc_a2 * unit_rc_a2;
            t8 *= t8;
            t8 *= t8;
            t8 *= t8;
            const float damp8 = 1.0f / fmaf(t8, 6.0f, 1.0f);
            const float r2_rc = r_rc * r_rc;
            const float r6_rc = r2_rc * r2_rc * r2_rc;
            const float r8_rc = r6_rc * r2_rc;
            const float x1 = 3.0f * c6 * r8_rc * fmaf(r2_rc, s8r42 * damp8 * fmaf(3.0f * alp8 * t8, damp8, -4.0f), s6 * damp6 * fmaf(alp6 * t6, damp6, -1.0f));

            const float vec[3] = { x1 * rij[0], x1 * rij[1], x1 * rij[2] };

            sigma_local[0] += vec[0] * rij[0];
            sigma_local[1] += vec[0] * rij[1];
            sigma_local[2] += vec[0] * rij[2];
            sigma_local[3] += vec[1] * rij[0];
            sigma_local[4] += vec[1] * rij[1];
            sigma_local[5] += vec[1] * rij[2];
            sigma_local[6] += vec[2] * rij[0];
            sigma_local[7] += vec[2] * rij[1];
            sigma_local[8] += vec[2] * rij[2];

            const float dc6_rest = 0.5f * r6_rc * fmaf(3.0f * r2_rc, s8r42 * damp8, s6 * damp6);
            disp_local -= dc6_rest * c6;
            dc6i_local_i += dc6_rest * dc6iji;
            dc6i_local_j += dc6_rest * dc6ijj;
        }
        atomicAdd(&dc6i[iat], dc6i_local_i);
        atomicAdd(&dc6i[jat], dc6i_local_j);
    } else {
        const int atomtype_i = atomtype[iat];
        const int atomtype_j = atomtype[jat];
        const float r0 = r0ab[atomtype_i][atomtype_j];
        const float r42 = r2r4[atomtype_i] * r2r4[atomtype_j];
        const float unit_a1 = (a1 * r0);
        const float unit_a2 = (a2 * r0);
        const float s8r42 = s8 * r42;

        for (int t = 0; t < n_tau; t++) {
            const float rij[3] = {
                x[jat * 3 + 0] - x[iat * 3 + 0] + tau_vdw[(tau_start + t) * 3 + 0],
                x[jat * 3 + 1] - x[iat * 3 + 1] + tau_vdw[(tau_start + t) * 3 + 1],
                x[jat * 3 + 2] - x[iat * 3 + 2] + tau_vdw[(tau_start + t) * 3 + 2]
            };
            const float r2 = lensq3(rij);
            if (r2 > rthr) continue;

            const float r_rc = rsqrtf(r2);
            float unit_rc_a1 = unit_a1 * r_rc;
            float t6 = unit_rc_a1 * unit_rc_a1;
            t6 *= unit_rc_a1;
            t6 *= t6;
            t6 *= unit_rc_a1;
            t6 *= t6;
            const float damp6 = 1.0f / fmaf(t6, 6.0f, 1.0f);
            float unit_rc_a2 = unit_a2 * r_rc;
            float t8 = unit_rc_a2 * unit_rc_a2;
            t8 *= t8;
            t8 *= t8;
            t8 *= t8;
            const float damp8 = 1.0f / fmaf(t8, 6.0f, 1.0f);
            const float r2_rc = r_rc * r_rc;
            const float r6_rc = r2_rc * r2_rc * r2_rc;
            const float r8_rc = r6_rc * r2_rc;
            const float x1 = 6.0f * c6 * r8_rc * fmaf(r2_rc, s8r42 * damp8 * fmaf(3.0f * alp8 * t8, damp8, -4.0f), s6 * damp6 * fmaf(alp6 * t6, damp6, -1.0f));

            const float vec[3] = { x1 * rij[0], x1 * rij[1], x1 * rij[2] };

            f_local[0] -= vec[0];
            f_local[1] -= vec[1];
            f_local[2] -= vec[2];

            sigma_local[0] += vec[0] * rij[0];
            sigma_local[1] += vec[0] * rij[1];
            sigma_local[2] += vec[0] * rij[2];
            sigma_local[3] += vec[1] * rij[0];
            sigma_local[4] += vec[1] * rij[1];
            sigma_local[5] += vec[1] * rij[2];
            sigma_local[6] += vec[2] * rij[0];
            sigma_local[7] += vec[2] * rij[1];
            sigma_local[8] += vec[2] * rij[2];

            const float dc6_rest = r6_rc * fmaf(3.0f * r2_rc, s8r42 * damp8, s6 * damp6);
            disp_local -= dc6_rest * c6;
            dc6i_local_i += dc6_rest * dc6iji;
            dc6i_local_j += dc6_rest * dc6ijj;
        }
        atomicAdd(&dc6i[iat], dc6i_local_i);
        atomicAdd(&dc6i[jat], dc6i_local_j);
        atomicAdd(&f[iat * 3 + 0], static_cast<double>(f_local[0]));
        atomicAdd(&f[iat * 3 + 1], static_cast<double>(f_local[1]));
        atomicAdd(&f[iat * 3 + 2], static_cast<double>(f_local[2]));
        atomicAdd(&f[jat * 3 + 0], static_cast<double>(-f_local[0]));
        atomicAdd(&f[jat * 3 + 1], static_cast<double>(-f_local[1]));
        atomicAdd(&f[jat * 3 + 2], static_cast<double>(-f_local[2]));
    }

    // Per-thread atomicAdd for sigma and disp (threads may span systems)
    atomicAdd(&disp[sys], static_cast<double>(disp_local));
    for (int i = 0; i < 9; i++) {
        if (sigma_local[i] != 0.0f) {
            atomicAdd(&sigma[sys * 9 + i], static_cast<double>(sigma_local[i]));
        }
    }
}

/* ---- Kernel 3b: Forces without dC6 (BJ damping) ---- */
__global__ void batch_kernel_forces_bj(
    int64_t total_pairs, int B,
    float rthr, float s6, float s8, float a1, float a2,
    float *r2r4, int *atomtype,
    float *x, float *c6_ij_tot, float *dc6_iji_tot, float *dc6_ijj_tot,
    double *dc6i, double *disp, double *f, double *sigma,
    int *atom_offset, int64_t *pair_offset,
    float *tau_vdw, int *tau_vdw_offset, int *center_tau_vdw
) {
    int64_t iter = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (iter >= total_pairs) return;

    int sys = find_system(iter, pair_offset, B);
    int64_t local_iter = iter - pair_offset[sys];
    int n_sys = atom_offset[sys + 1] - atom_offset[sys];
    int64_t maxij_sys = static_cast<int64_t>(n_sys) * (n_sys + 1) / 2;
    if (local_iter >= maxij_sys) return;

    int iat_local, jat_local;
    ij_at_linij(local_iter, iat_local, jat_local);

    int iat = iat_local + atom_offset[sys];
    int jat = jat_local + atom_offset[sys];

    int tau_start = tau_vdw_offset[sys];
    int n_tau = tau_vdw_offset[sys + 1] - tau_start;
    int center_t = center_tau_vdw[sys];

    float f_local[3] = { 0.0f };
    float dc6i_local_i = 0.0f;
    float dc6i_local_j = 0.0f;
    float sigma_local[9] = { 0.0f };
    float disp_local = 0.0f;

    const float c6 = c6_ij_tot[iter];
    const float dc6iji = dc6_iji_tot[iter];
    const float dc6ijj = dc6_ijj_tot[iter];

    if (iat == jat) {
        const float unit_r2r4 = r2r4[atomtype[iat]];
        const float r42x3 = unit_r2r4 * unit_r2r4 * 3.0f;
        const float R0 = fmaf(a1, sqrtf(r42x3), a2);
        const float R0_2 = R0 * R0;
        const float R0_6 = R0_2 * R0_2 * R0_2;
        const float R0_8 = R0_6 * R0_2;
        const float s8r42x3 = s8 * r42x3;

        for (int t = 0; t < n_tau; t++) {
            if (t == center_t) continue;
            const float rij[3] = {
                tau_vdw[(tau_start + t) * 3 + 0],
                tau_vdw[(tau_start + t) * 3 + 1],
                tau_vdw[(tau_start + t) * 3 + 2]
            };
            const float r2 = lensq3(rij);
            if (r2 > rthr) continue;

            const float r = sqrtf(r2);
            const float r5 = r2 * r2 * r;
            const float r7 = r5 * r2;
            const float t6_rc = 1.0f / fmaf(r5, r, R0_6);
            const float t8_rc = 1.0f / fmaf(r7, r, R0_8);
            const float t6_sqrc = t6_rc * t6_rc;
            const float t8_sqrc = t8_rc * t8_rc;
            const float x1 = -c6 * fmaf(4.0f * s8r42x3 * r7, t8_sqrc, 3.0f * s6 * r5 * t6_sqrc);

            const float r_rc = 1.0f / r;
            const float vec[3] = {
                x1 * rij[0] * r_rc,
                x1 * rij[1] * r_rc,
                x1 * rij[2] * r_rc
            };

            sigma_local[0] += vec[0] * rij[0];
            sigma_local[1] += vec[0] * rij[1];
            sigma_local[2] += vec[0] * rij[2];
            sigma_local[3] += vec[1] * rij[0];
            sigma_local[4] += vec[1] * rij[1];
            sigma_local[5] += vec[1] * rij[2];
            sigma_local[6] += vec[2] * rij[0];
            sigma_local[7] += vec[2] * rij[1];
            sigma_local[8] += vec[2] * rij[2];

            const float dc6_rest = 0.5f * fmaf(s8r42x3, t8_rc, s6 * t6_rc);
            disp_local -= dc6_rest * c6;
            dc6i_local_i += dc6_rest * dc6iji;
            dc6i_local_j += dc6_rest * dc6ijj;
        }
        atomicAdd(&dc6i[iat], dc6i_local_i);
        atomicAdd(&dc6i[jat], dc6i_local_j);
    } else {
        const float r42x3 = r2r4[atomtype[iat]] * r2r4[atomtype[jat]] * 3.0f;
        const float R0 = fmaf(a1, sqrtf(r42x3), a2);
        const float R0_2 = R0 * R0;
        const float R0_6 = R0_2 * R0_2 * R0_2;
        const float R0_8 = R0_6 * R0_2;
        const float s8r42x3 = s8 * r42x3;

        for (int t = 0; t < n_tau; t++) {
            const float rij[3] = {
                x[jat * 3 + 0] - x[iat * 3 + 0] + tau_vdw[(tau_start + t) * 3 + 0],
                x[jat * 3 + 1] - x[iat * 3 + 1] + tau_vdw[(tau_start + t) * 3 + 1],
                x[jat * 3 + 2] - x[iat * 3 + 2] + tau_vdw[(tau_start + t) * 3 + 2]
            };
            const float r2 = lensq3(rij);
            if (r2 > rthr) continue;

            const float r = sqrtf(r2);
            const float r5 = r2 * r2 * r;
            const float r7 = r5 * r2;
            const float t6_rc = 1.0f / fmaf(r5, r, R0_6);
            const float t8_rc = 1.0f / fmaf(r7, r, R0_8);
            const float t6_sqrc = t6_rc * t6_rc;
            const float t8_sqrc = t8_rc * t8_rc;
            const float x1 = -c6 * fmaf(8.0f * s8r42x3 * r7, t8_sqrc, 6.0f * s6 * r5 * t6_sqrc);

            const float r_rc = 1.0f / r;
            const float vec[3] = {
                x1 * rij[0] * r_rc,
                x1 * rij[1] * r_rc,
                x1 * rij[2] * r_rc
            };

            f_local[0] -= vec[0];
            f_local[1] -= vec[1];
            f_local[2] -= vec[2];

            sigma_local[0] += vec[0] * rij[0];
            sigma_local[1] += vec[0] * rij[1];
            sigma_local[2] += vec[0] * rij[2];
            sigma_local[3] += vec[1] * rij[0];
            sigma_local[4] += vec[1] * rij[1];
            sigma_local[5] += vec[1] * rij[2];
            sigma_local[6] += vec[2] * rij[0];
            sigma_local[7] += vec[2] * rij[1];
            sigma_local[8] += vec[2] * rij[2];

            const float dc6_rest = fmaf(s8r42x3, t8_rc, s6 * t6_rc);
            disp_local -= dc6_rest * c6;
            dc6i_local_i += dc6_rest * dc6iji;
            dc6i_local_j += dc6_rest * dc6ijj;
        }
        atomicAdd(&dc6i[iat], dc6i_local_i);
        atomicAdd(&dc6i[jat], dc6i_local_j);
        atomicAdd(&f[iat * 3 + 0], static_cast<double>(f_local[0]));
        atomicAdd(&f[iat * 3 + 1], static_cast<double>(f_local[1]));
        atomicAdd(&f[iat * 3 + 2], static_cast<double>(f_local[2]));
        atomicAdd(&f[jat * 3 + 0], static_cast<double>(-f_local[0]));
        atomicAdd(&f[jat * 3 + 1], static_cast<double>(-f_local[1]));
        atomicAdd(&f[jat * 3 + 2], static_cast<double>(-f_local[2]));
    }

    atomicAdd(&disp[sys], static_cast<double>(disp_local));
    for (int i = 0; i < 9; i++) {
        if (sigma_local[i] != 0.0f) {
            atomicAdd(&sigma[sys * 9 + i], static_cast<double>(sigma_local[i]));
        }
    }
}

/* ---- Kernel 4: Forces with dC6 (CN-dependent part) ---- */
__global__ void batch_kernel_forces_with_dC6(
    int64_t total_pairs, int B,
    float cnthr, float K1,
    double *dc6i, float *rcov, int *atomtype,
    float *x, double *f, double *sigma,
    int *atom_offset, int64_t *pair_offset,
    float *tau_cn, int *tau_cn_offset, int *center_tau_cn
) {
    int64_t iter = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (iter >= total_pairs) return;

    int sys = find_system(iter, pair_offset, B);
    int64_t local_iter = iter - pair_offset[sys];
    int n_sys = atom_offset[sys + 1] - atom_offset[sys];
    int64_t maxij_sys = static_cast<int64_t>(n_sys) * (n_sys + 1) / 2;
    if (local_iter >= maxij_sys) return;

    int iat_local, jat_local;
    ij_at_linij(local_iter, iat_local, jat_local);

    int iat = iat_local + atom_offset[sys];
    int jat = jat_local + atom_offset[sys];

    int tau_start = tau_cn_offset[sys];
    int n_tau = tau_cn_offset[sys + 1] - tau_start;
    int center_t = center_tau_cn[sys];

    float f_local[3] = { 0.0f };
    float sigma_local[9] = { 0.0f };

    if (iat == jat) {
        const float rcov_sum = rcov[atomtype[iat]] * 2.0f;
        const float dc6i_sum = dc6i[iat];

        for (int t = 0; t < n_tau; t++) {
            if (t == center_t) continue;
            const float rij[3] = {
                tau_cn[(tau_start + t) * 3 + 0],
                tau_cn[(tau_start + t) * 3 + 1],
                tau_cn[(tau_start + t) * 3 + 2]
            };
            const float r2 = lensq3(rij);
            if (r2 >= cnthr) continue;

            const float r_rc = rsqrtf(r2);
            const float expterm = expf(-K1 * (rcov_sum * r_rc - 1.0f));
            const float unit_rc = 1.0f / (r2 * (expterm + 1.0f) * (expterm + 1.0f));
            const float dcnn = -K1 * rcov_sum * expterm * unit_rc;
            const float x1 = dcnn * dc6i_sum;

            const float vec[3] = {
                x1 * rij[0] * r_rc,
                x1 * rij[1] * r_rc,
                x1 * rij[2] * r_rc
            };

            sigma_local[0] += vec[0] * rij[0];
            sigma_local[1] += vec[0] * rij[1];
            sigma_local[2] += vec[0] * rij[2];
            sigma_local[3] += vec[1] * rij[0];
            sigma_local[4] += vec[1] * rij[1];
            sigma_local[5] += vec[1] * rij[2];
            sigma_local[6] += vec[2] * rij[0];
            sigma_local[7] += vec[2] * rij[1];
            sigma_local[8] += vec[2] * rij[2];
        }
    } else {
        const float rcov_sum = rcov[atomtype[iat]] + rcov[atomtype[jat]];
        const float dc6i_sum = dc6i[iat] + dc6i[jat];

        for (int t = 0; t < n_tau; t++) {
            const float rij[3] = {
                x[jat * 3 + 0] - x[iat * 3 + 0] + tau_cn[(tau_start + t) * 3 + 0],
                x[jat * 3 + 1] - x[iat * 3 + 1] + tau_cn[(tau_start + t) * 3 + 1],
                x[jat * 3 + 2] - x[iat * 3 + 2] + tau_cn[(tau_start + t) * 3 + 2]
            };
            const float r2 = lensq3(rij);
            if (r2 >= cnthr) continue;

            const float r_rc = rsqrtf(r2);
            const float expterm = expf(-K1 * (rcov_sum * r_rc - 1.0f));
            const float unit_rc = 1.0f / (r2 * (expterm + 1.0f) * (expterm + 1.0f));
            const float dcnn = -K1 * rcov_sum * expterm * unit_rc;
            const float x1 = dcnn * dc6i_sum;

            const float vec[3] = {
                x1 * rij[0] * r_rc,
                x1 * rij[1] * r_rc,
                x1 * rij[2] * r_rc
            };

            f_local[0] -= vec[0];
            f_local[1] -= vec[1];
            f_local[2] -= vec[2];

            sigma_local[0] += vec[0] * rij[0];
            sigma_local[1] += vec[0] * rij[1];
            sigma_local[2] += vec[0] * rij[2];
            sigma_local[3] += vec[1] * rij[0];
            sigma_local[4] += vec[1] * rij[1];
            sigma_local[5] += vec[1] * rij[2];
            sigma_local[6] += vec[2] * rij[0];
            sigma_local[7] += vec[2] * rij[1];
            sigma_local[8] += vec[2] * rij[2];
        }
        atomicAdd(&f[iat * 3 + 0], static_cast<double>(f_local[0]));
        atomicAdd(&f[iat * 3 + 1], static_cast<double>(f_local[1]));
        atomicAdd(&f[iat * 3 + 2], static_cast<double>(f_local[2]));
        atomicAdd(&f[jat * 3 + 0], static_cast<double>(-f_local[0]));
        atomicAdd(&f[jat * 3 + 1], static_cast<double>(-f_local[1]));
        atomicAdd(&f[jat * 3 + 2], static_cast<double>(-f_local[2]));
    }

    for (int i = 0; i < 9; i++) {
        if (sigma_local[i] != 0.0f) {
            atomicAdd(&sigma[sys * 9 + i], static_cast<double>(sigma_local[i]));
        }
    }
}

/* =========================================================================
   Compute (main entry point)
   ========================================================================= */

void BatchPairD3::compute_async(int B,
                                 const int* natoms_each,
                                 const int* h_atomtype,
                                 const double* h_x_flat,
                                 const double* h_cells,
                                 const int* h_pbc) {
    // ---- 1. Compute offsets and sizes ----
    // Temporary CPU storage for per-system info
    std::vector<int> rep_vdw_all(B * 3), rep_cn_all(B * 3);
    std::vector<int> n_tau_vdw(B), n_tau_cn(B);
    std::vector<double> lat_v_all(B * 9); // [B][3][3] column-major: lat_v_all[s*9 + col*3 + row]

    int N_total = 0;
    int64_t P_total = 0;
    int T_vdw_total = 0, T_cn_total = 0;

    for (int s = 0; s < B; s++) {
        N_total += natoms_each[s];
        int64_t n = natoms_each[s];
        P_total += n * (n + 1) / 2;

        // Convert cells[s*9..] (row-major: row0=a, row1=b, row2=c) to lat_v (column-major in Bohr)
        // cells is [a0 a1 a2 | b0 b1 b2 | c0 c1 c2] in Angstrom
        // lat_v[col][row]: col0 = a-vector, col1 = b-vector, col2 = c-vector
        double lv[3][3];
        for (int col = 0; col < 3; col++) {
            for (int row = 0; row < 3; row++) {
                lv[col][row] = h_cells[s * 9 + col * 3 + row] / AU_TO_ANG;
            }
        }
        for (int i = 0; i < 9; i++) lat_v_all[s * 9 + i] = ((double*)lv)[i];

        int pbc_flags[3] = { h_pbc[s * 3], h_pbc[s * 3 + 1], h_pbc[s * 3 + 2] };
        compute_lattice_reps(lv, pbc_flags, rthr, &rep_vdw_all[s * 3]);
        compute_lattice_reps(lv, pbc_flags, cnthr, &rep_cn_all[s * 3]);

        n_tau_vdw[s] = (2 * rep_vdw_all[s * 3] + 1) *
                        (2 * rep_vdw_all[s * 3 + 1] + 1) *
                        (2 * rep_vdw_all[s * 3 + 2] + 1);
        n_tau_cn[s]  = (2 * rep_cn_all[s * 3] + 1) *
                        (2 * rep_cn_all[s * 3 + 1] + 1) *
                        (2 * rep_cn_all[s * 3 + 2] + 1);
        T_vdw_total += n_tau_vdw[s];
        T_cn_total  += n_tau_cn[s];
    }

    // ---- 2. Ensure GPU memory ----
    ensure_capacity(B, N_total, P_total, T_vdw_total, T_cn_total);

    // ---- 3. Fill offset arrays ----
    atom_offset[0] = 0;
    pair_offset[0] = 0;
    tau_vdw_offset[0] = 0;
    tau_cn_offset[0] = 0;
    for (int s = 0; s < B; s++) {
        atom_offset[s + 1] = atom_offset[s] + natoms_each[s];
        int64_t n = natoms_each[s];
        pair_offset[s + 1] = pair_offset[s] + n * (n + 1) / 2;
        tau_vdw_offset[s + 1] = tau_vdw_offset[s] + n_tau_vdw[s];
        tau_cn_offset[s + 1]  = tau_cn_offset[s]  + n_tau_cn[s];
    }

    // ---- 4. Fill tau arrays and center indices ----
    for (int s = 0; s < B; s++) {
        double lv[3][3];
        for (int i = 0; i < 9; i++) ((double*)lv)[i] = lat_v_all[s * 9 + i];

        int center_vdw, center_cn;
        compute_tau_flat(lv, &rep_vdw_all[s * 3],
                         &d_tau_vdw[tau_vdw_offset[s] * 3],
                         n_tau_vdw[s], center_vdw);
        compute_tau_flat(lv, &rep_cn_all[s * 3],
                         &d_tau_cn[tau_cn_offset[s] * 3],
                         n_tau_cn[s], center_cn);
        center_tau_vdw[s] = center_vdw;
        center_tau_cn[s]  = center_cn;
    }

    // ---- 5. Load atom positions (wrap into unit cell, convert to Bohr) and types ----
    for (int s = 0; s < B; s++) {
        int n = natoms_each[s];
        int a_off = atom_offset[s];
        double lv[3][3];
        for (int i = 0; i < 9; i++) ((double*)lv)[i] = lat_v_all[s * 9 + i];

        // lat[row][col] for inversion (column vectors are lv[col][row])
        double lat[3][3];
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                lat[row][col] = lv[col][row];

        double det = lat[0][0] * lat[1][1] * lat[2][2]
                   + lat[0][1] * lat[1][2] * lat[2][0]
                   + lat[0][2] * lat[1][0] * lat[2][1]
                   - lat[0][2] * lat[1][1] * lat[2][0]
                   - lat[0][1] * lat[1][0] * lat[2][2]
                   - lat[0][0] * lat[1][2] * lat[2][1];

        double lat_inv[3][3];
        lat_inv[0][0] = (lat[1][1] * lat[2][2] - lat[1][2] * lat[2][1]) / det;
        lat_inv[1][0] = (lat[1][2] * lat[2][0] - lat[1][0] * lat[2][2]) / det;
        lat_inv[2][0] = (lat[1][0] * lat[2][1] - lat[1][1] * lat[2][0]) / det;
        lat_inv[0][1] = (lat[0][2] * lat[2][1] - lat[0][1] * lat[2][2]) / det;
        lat_inv[1][1] = (lat[0][0] * lat[2][2] - lat[0][2] * lat[2][0]) / det;
        lat_inv[2][1] = (lat[0][1] * lat[2][0] - lat[0][0] * lat[2][1]) / det;
        lat_inv[0][2] = (lat[0][1] * lat[1][2] - lat[0][2] * lat[1][1]) / det;
        lat_inv[1][2] = (lat[0][2] * lat[1][0] - lat[0][0] * lat[1][2]) / det;
        lat_inv[2][2] = (lat[0][0] * lat[1][1] - lat[0][1] * lat[1][0]) / det;

        for (int iat = 0; iat < n; iat++) {
            // Input positions are in Angstrom; convert to Bohr for fractional coords
            double a[3];
            for (int i = 0; i < 3; i++) {
                a[i] = lat_inv[i][0] * h_x_flat[(a_off + iat) * 3 + 0] / AU_TO_ANG +
                       lat_inv[i][1] * h_x_flat[(a_off + iat) * 3 + 1] / AU_TO_ANG +
                       lat_inv[i][2] * h_x_flat[(a_off + iat) * 3 + 2] / AU_TO_ANG;
                a[i] -= floor(a[i]);
            }
            for (int i = 0; i < 3; i++) {
                d_x[(a_off + iat) * 3 + i] = static_cast<float>(
                    lat[i][0] * a[0] + lat[i][1] * a[1] + lat[i][2] * a[2]);
            }
        }
    }

    // Copy atom types
    memcpy(d_atomtype, h_atomtype, N_total * sizeof(int));

    // ---- 6. Zero output arrays ----
    cudaMemsetAsync(d_cn,   0, N_total * sizeof(double), stream);
    cudaMemsetAsync(d_dc6i, 0, N_total * sizeof(double), stream);
    cudaMemsetAsync(d_f,    0, N_total * 3 * sizeof(double), stream);
    cudaMemsetAsync(d_disp, 0, B * sizeof(double), stream);
    cudaMemsetAsync(d_sigma, 0, B * 9 * sizeof(double), stream);

    // ---- 7. Launch kernels ----
    const int threadsPerBlock = 128;

    // Kernel 1: Coordination numbers
    {
        int64_t total = pair_offset[B];
        int blocks = static_cast<int>((total + threadsPerBlock - 1) / threadsPerBlock);
        batch_kernel_get_coordination_number<<<blocks, threadsPerBlock, 0, stream>>>(
            total, B, cnthr, K1,
            rcov, d_atomtype, d_x, d_cn,
            atom_offset, pair_offset,
            d_tau_cn, tau_cn_offset, center_tau_cn
        );
    }

    // Kernel 2: dC6/dCN
    {
        int64_t total = pair_offset[B];
        int blocks = static_cast<int>((total + threadsPerBlock - 1) / threadsPerBlock);
        batch_kernel_get_dC6_dCNij<<<blocks, threadsPerBlock, 0, stream>>>(
            total, B, K3,
            d_cn, mxc, c6ab, d_atomtype,
            d_c6_ij_tot, d_dc6_iji_tot, d_dc6_ijj_tot,
            atom_offset, pair_offset
        );
    }

    // Kernel 3: Forces without dC6 (damping-dependent)
    {
        int64_t total = pair_offset[B];
        int blocks = static_cast<int>((total + threadsPerBlock - 1) / threadsPerBlock);
        if (damping == 0) { // zero
            batch_kernel_forces_zero<<<blocks, threadsPerBlock, 0, stream>>>(
                total, B, rthr, s6, s8, a1, a2, alp6, alp8,
                r2r4, r0ab, d_atomtype,
                d_x, d_c6_ij_tot, d_dc6_iji_tot, d_dc6_ijj_tot,
                d_dc6i, d_disp, d_f, d_sigma,
                atom_offset, pair_offset,
                d_tau_vdw, tau_vdw_offset, center_tau_vdw
            );
        } else if (damping == 1) { // bj
            batch_kernel_forces_bj<<<blocks, threadsPerBlock, 0, stream>>>(
                total, B, rthr, s6, s8, a1, a2,
                r2r4, d_atomtype,
                d_x, d_c6_ij_tot, d_dc6_iji_tot, d_dc6_ijj_tot,
                d_dc6i, d_disp, d_f, d_sigma,
                atom_offset, pair_offset,
                d_tau_vdw, tau_vdw_offset, center_tau_vdw
            );
        } else {
            // zerom/bjm: stubs (not yet implemented)
            fprintf(stderr, "Error: zerom/bjm damping not implemented for batch D3\n");
            exit(EXIT_FAILURE);
        }
    }

    // Kernel 4: Forces with dC6 (CN-dependent correction)
    {
        int64_t total = pair_offset[B];
        int blocks = static_cast<int>((total + threadsPerBlock - 1) / threadsPerBlock);
        batch_kernel_forces_with_dC6<<<blocks, threadsPerBlock, 0, stream>>>(
            total, B, cnthr, K1,
            d_dc6i, rcov, d_atomtype,
            d_x, d_f, d_sigma,
            atom_offset, pair_offset,
            d_tau_cn, tau_cn_offset, center_tau_cn
        );
    }
}

void BatchPairD3::sync(int B,
                        const int* natoms_each,
                        double* energy_out,
                        double* forces_out,
                        double* stress_out) {
    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR_STREAM(stream);

    int N_total = atom_offset[B];

    // Convert energy: AU -> eV
    for (int s = 0; s < B; s++) {
        energy_out[s] = d_disp[s] * AU_TO_EV;
    }

    // Convert forces: AU -> eV/Ang
    const double f_conv = AU_TO_EV / AU_TO_ANG;
    for (int i = 0; i < N_total * 3; i++) {
        forces_out[i] = d_f[i] * f_conv;
    }

    // Convert stress: sigma[sys*9+..] -> Voigt (xx,yy,zz,xy,xz,yz) in eV
    // Output as full 3x3 for TorchSim: stress_out[s*9 + row*3 + col]
    for (int s = 0; s < B; s++) {
        for (int i = 0; i < 9; i++) {
            stress_out[s * 9 + i] = d_sigma[s * 9 + i] * AU_TO_EV;
        }
    }
}

void BatchPairD3::compute(int B,
                           const int* natoms_each,
                           const int* atomtype,
                           const double* x_flat,
                           const double* cells,
                           const int* pbc,
                           double* energy_out,
                           double* forces_out,
                           double* stress_out) {
    compute_async(B, natoms_each, atomtype, x_flat, cells, pbc);
    sync(B, natoms_each, energy_out, forces_out, stress_out);
}

/* =========================================================================
   C API
   ========================================================================= */

extern "C" {
    BatchPairD3* batch_d3_init() {
        return new BatchPairD3();
    }

    void batch_d3_settings(BatchPairD3* obj,
                           double rthr, double cnthr,
                           const char* damp_name, const char* func_name) {
        obj->settings(rthr, cnthr, damp_name, func_name);
    }

    void batch_d3_coeff(BatchPairD3* obj,
                        int* atomic_numbers, int ntypes) {
        obj->coeff(atomic_numbers, ntypes);
    }

    void batch_d3_compute(BatchPairD3* obj,
                          int B,
                          const int* natoms_each,
                          const int* atomtype,
                          const double* x_flat,
                          const double* cells,
                          const int* pbc,
                          double* energy_out,
                          double* forces_out,
                          double* stress_out) {
        obj->compute(B, natoms_each, atomtype, x_flat, cells, pbc,
                     energy_out, forces_out, stress_out);
    }

    void batch_d3_compute_async(BatchPairD3* obj,
                                int B,
                                const int* natoms_each,
                                const int* atomtype,
                                const double* x_flat,
                                const double* cells,
                                const int* pbc) {
        obj->compute_async(B, natoms_each, atomtype, x_flat, cells, pbc);
    }

    void batch_d3_sync(BatchPairD3* obj,
                       int B,
                       const int* natoms_each,
                       double* energy_out,
                       double* forces_out,
                       double* stress_out) {
        obj->sync(B, natoms_each, energy_out, forces_out, stress_out);
    }

    void batch_d3_fin(BatchPairD3* obj) {
        delete obj;
    }
}

int main() {
    return 0;
}
