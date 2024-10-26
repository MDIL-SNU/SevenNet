/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Gijin Kim, Hyungmin An (SNU)
------------------------------------------------------------------------- */

#include "pair_d3.h"
#include <cuda_runtime.h>

using namespace LAMMPS_NS;

/* ------- Macros for CUDA error handling ------- */
#define START_CUDA_TIMER()    \
    cudaEvent_t start, stop;  \
    cudaEventCreate(&start);  \
    cudaEventCreate(&stop);   \
    cudaEventRecord(start);

#define STOP_CUDA_TIMER(tag)                           \
    cudaEventRecord(stop);                             \
    cudaEventSynchronize(stop);                        \
    float msec = 0;                                    \
    cudaEventElapsedTime(&msec, start, stop);          \
    printf("Elapsed time for %s: %f ms\n", tag, msec); \
    cudaEventDestroy(start);                           \
    cudaEventDestroy(stop);

#define CHECK_CUDA(call) do {                                            \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
} while (0)

#define CHECK_CUDA_ERROR() do {                                          \
    cudaDeviceSynchronize();                                             \
    cudaError_t status_ = cudaGetLastError();                            \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
} while (0)

#define CHECK_CUDA_DEVICES() do {                                              \
    int deviceCount = 0;                                                       \
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) { \
        error->all(FLERR, "No CUDA devices found. Exiting...");                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)
/* ------- Macros for CUDA error handling ------- */

/* ------- Math functions for CUDA compatibility ------- */
int *atomtype;
double *dispall;

inline __host__ __device__ void ij_at_linij(int linij, int &i, int &j) {
    i = static_cast<int>((sqrt(1 + 8 * linij) - 1) / 2);
    j = linij - i * (i + 1) / 2;
} // unroll the triangular loop

inline __host__ __device__ float lensq3(const float *v)
{
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
} // from MathExtra::lensq3
/* ------- Math functions for CUDA compatibility ------- */

/* ----------------------------------------------------------------------
   Constructor (Required)
------------------------------------------------------------------------- */

PairD3::PairD3(LAMMPS* lmp) : Pair(lmp) {
    single_enable = 0;      // potential is not pair-wise additive.
    restartinfo = 0;        // Many-body potentials are usually not
                            // written to binary restart files.
    one_coeff = 1;          // Many-body potnetials typically read all
                            // parameters from a file, so only one
                            // pair_coeff statement is needed.
    manybody_flag = 1;
    no_virial_fdotr_compute = 1;
}

/* ----------------------------------------------------------------------
   Destructor (Required)
------------------------------------------------------------------------- */

PairD3::~PairD3() {
    if (allocated) {

        int n = atom->natoms;
        int np1 = atom->ntypes + 1;
        int vdw_range_x = 2 * rep_vdw[0] + 1;
        int vdw_range_y = 2 * rep_vdw[1] + 1;
        int vdw_range_z = 2 * rep_vdw[2] + 1;
        int cn_range_x  = 2 * rep_cn[0] + 1;
        int cn_range_y  = 2 * rep_cn[1] + 1;
        int cn_range_z  = 2 * rep_cn[2] + 1;

        for (int i = 0; i < np1; i++) { cudaFree(setflag[i]); }; cudaFree(setflag);
        for (int i = 0; i < np1; i++) { cudaFree(cutsq[i]); }; cudaFree(cutsq);
        cudaFree(r2r4);
        cudaFree(rcov);
        cudaFree(mxc);
        for (int i = 0; i < np1; i++) { cudaFree(r0ab[i]); }; cudaFree(r0ab);
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
        cudaFree(c6ab);

        cudaFree(lat_v_1);
        cudaFree(lat_v_2);
        cudaFree(lat_v_3);

        cudaFree(rep_vdw);
        cudaFree(rep_cn);
        cudaFree(cn);
        for (int i = 0; i < n; i++) { cudaFree(x[i]); }; cudaFree(x);

        cudaFree(dc6i);
        for (int i = 0; i < n; i++) { cudaFree(f[i]); }; cudaFree(f);

        for (int i = 0; i < 3; i++) { cudaFree(sigma[i]); }; cudaFree(sigma);

        cudaFree(dc6_iji_tot);
        cudaFree(dc6_ijj_tot);
        cudaFree(c6_ij_tot);

        for (int i = 0; i < vdw_range_x; i++) {
            for (int j = 0; j < vdw_range_y; j++) {
                for (int k = 0; k < vdw_range_z; k++) {
                    cudaFree(tau_vdw[i][j][k]);
                }
                cudaFree(tau_vdw[i][j]);
            }
            cudaFree(tau_vdw[i]);
        }
        cudaFree(tau_vdw);
        for (int i = 0; i < cn_range_x; i++) {
            for (int j = 0; j < cn_range_y; j++) {
                for (int k = 0; k < cn_range_z; k++) {
                    cudaFree(tau_cn[i][j][k]);
                }
                cudaFree(tau_cn[i][j]);
            }
            cudaFree(tau_cn[i]);
        }
        cudaFree(tau_cn);

        cudaFree(tau_idx_vdw);
        cudaFree(tau_idx_cn);

        cudaFree(atomtype);
        cudaFree(dispall);

        //CHECK_CUDA_ERROR();
    }
}

/* ----------------------------------------------------------------------
   Allocate all arrays (Required)
------------------------------------------------------------------------- */

void PairD3::allocate() {
    CHECK_CUDA_DEVICES();
    allocated = 1;

    /* atom->ntypes : # of elements; element index starts from 1 */
    int n = atom->natoms;
    int np1 = atom->ntypes + 1;
    n_save = n;

    cudaMallocManaged(&setflag, np1 * sizeof(int*)); for (int i = 0; i < np1; i++) { cudaMallocManaged(&setflag[i], np1 * sizeof(int)); }
    cudaMallocManaged(&cutsq, np1 * sizeof(double*)); for (int i = 0; i < np1; i++) { cudaMallocManaged(&cutsq[i], np1 * sizeof(double)); }
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

    cudaMallocManaged(&lat_v_1, 3 * sizeof(float));
    cudaMallocManaged(&lat_v_2, 3 * sizeof(float));
    cudaMallocManaged(&lat_v_3, 3 * sizeof(float));
    cudaMallocManaged(&rep_vdw, 3 * sizeof(int));
    cudaMallocManaged(&rep_cn,  3 * sizeof(int));
    cudaMallocManaged(&sigma,   3 * sizeof(double*)); for (int i = 0; i < 3; i++) { cudaMallocManaged(&sigma[i], 3 * sizeof(double)); }

    cudaMallocManaged(&cn, n * sizeof(double));
    cudaMallocManaged(&x, n * sizeof(float*)); for (int i = 0; i < n; i++) { cudaMallocManaged(&x[i], 3 * sizeof(float)); }
    cudaMallocManaged(&dc6i, n * sizeof(double));
    cudaMallocManaged(&f, n * sizeof(double*)); for (int i = 0; i < n; i++) { cudaMallocManaged(&f[i], 3 * sizeof(double)); }

    // Initialization (by function)
    set_lattice_vectors();

    // Initialization
    for (int i = 1; i < np1; i++) {
        for (int j = 1; j < np1; j++) {
            setflag[i][j] = 0;
        }
    }

    for (int idx1 = 0; idx1 < np1; idx1++) {
        for (int idx2 = 0; idx2 < np1; idx2++) {
            for (int idx3 = 0; idx3 < MAXC; idx3++) {
                for (int idx4 = 0; idx4 < MAXC; idx4++) {
                    for (int idx5 = 0; idx5 < 3; idx5++) {
                        c6ab[idx1][idx2][idx3][idx4][idx5] = -1;
                    }
                }
            }
        }
    }

    int n_ij_combination = n * (n + 1) / 2;
    cudaMallocManaged(&dc6_iji_tot, n_ij_combination * sizeof(float));
    cudaMallocManaged(&dc6_ijj_tot, n_ij_combination * sizeof(float));
    cudaMallocManaged(&c6_ij_tot,   n_ij_combination * sizeof(float));

    cudaMallocManaged(&atomtype, n * sizeof(int));
    cudaMallocManaged(&dispall, sizeof(double));

    //CHECK_CUDA_ERROR();
}

/* ----------------------------------------------------------------------
   Settings: read from pair_style (Required)
             pair_style   d3 rthr cn_thr damping_type
------------------------------------------------------------------------- */

void PairD3::settings(int narg, char **arg) {
    if (narg != 4) {
        error->all(FLERR,
                "Pair_style d3 needs Three arguments:\n"
                "\t rthr : threshold for dispersion interaction\n"
                "\t cn_thr : threshold for coordination number calculation\n"
                "\t damping_type : type of damping function\n"
                "\t functional_name : name of the functional\n"
                );
    }
    rthr   = utils::numeric(FLERR, arg[0], false, lmp);
    cn_thr = utils::numeric(FLERR, arg[1], false, lmp);

    std::unordered_map<std::string, int> commandMap = {
        { "damp_zero", 1}, { "damp_bj", 2 },
        { "damp_zerom", 3 }, { "damp_bjm", 4 },
    };

    int commandCode = commandMap[arg[2]];
    switch (commandCode) {
    case 1: damping_type = 1; break;
    case 2: damping_type = 2; break;
    case 3: damping_type = 3; break;
    case 4: damping_type = 4; break;
    default:
        error->all(FLERR,
                "Unknown damping type\n"
                "\t\t'damp_zero',\n"
                "\t\t'damp_bj',\n"
                "\t\t'damp_zerom',\n"
                "\t\t'damp_bjm'\n"
                );
        break;
    }

    // read functional parameters
    setfuncpar(arg[3]);
}


/* ----------------------------------------------------------------------
   finds atomic number (used in PairD3::coeff)
------------------------------------------------------------------------- */

int PairD3::find_atomic_number(std::string& key) {
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
            int atomic_number = i + 1;
            return atomic_number;
        }
    }

    // if not the case
    return -1;
}

/* ----------------------------------------------------------------------
   Check whether an integer value in an integer array (used in PairD3::coeff)
------------------------------------------------------------------------- */

int PairD3::is_int_in_array(int arr[], int size, int value) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == value) { return i; } // returns the index
    }
    return -1;
}

/* ----------------------------------------------------------------------
   Read r0ab values from r0ab.csv (used in PairD3::coeff)
------------------------------------------------------------------------- */

void PairD3::read_r0ab(int* atomic_numbers, int ntypes) {
    const double r0ab_table[94][94] = R0AB_TABLE;

    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            r0ab[i][j] = r0ab_table[atomic_numbers[i-1]-1][atomic_numbers[j-1]-1] / AU_TO_ANG;
        }
    }

}

/* ----------------------------------------------------------------------
   Get atom pair indices and grid indices (used in PairD3::read_c6ab)
------------------------------------------------------------------------- */

void PairD3::get_limit_in_pars_array(int& idx_atom_1, int& idx_atom_2, int& idx_i, int& idx_j) {
    idx_i = 1;
    idx_j = 1;
    int shift = 100;

    while (idx_atom_1 > shift) {
        idx_atom_1 -= shift;
        idx_i++;
    }

    while (idx_atom_2 > shift) {
        idx_atom_2 -= shift;
        idx_j++;
    }
}

/* ----------------------------------------------------------------------
   Read c6ab values from c6ab.csv (used in PairD3::coeff)
------------------------------------------------------------------------- */

void PairD3::read_c6ab(int* atomic_numbers, int ntypes) {
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
   Set functional parameters (used in PairD3::settings)
------------------------------------------------------------------------- */

void PairD3::setfuncpar(char* functional_name) {
    // set parameters for the given functionals
    int zero_damping = 1;
    int bj_damping = 2;
    int zero_damping_modified = 3;
    int bj_damping_modified = 4;

    if (damping_type == zero_damping) {
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

        int commandCode = commandMap[functional_name];
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
            error->all(FLERR, "Functional name unknown");
            break;
        }

    } else if (damping_type == bj_damping) {
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
            {"pbeh-3c", 52}
        };

        int commandCode = commandMap[functional_name];
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
            default:
                error->all(FLERR, "Functional name unknown");
                break;
        }
    } else if (damping_type == zero_damping_modified) {
        s6 = 1.0;
        alp = 14.0;

        std::unordered_map<std::string, int> commandMap = {
            {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
            {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
        };

        int commandCode = commandMap[functional_name];
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
                error->all(FLERR, "Functional name unknown");
                break;
        }
    } else if (damping_type == bj_damping_modified) {
        // BJ damping
        s6 = 1.0;
        alp = 14.0;

        std::unordered_map<std::string, int> commandMap = {
            {"b2-plyp", 1}, {"b3-lyp", 2}, {"b97-d", 3}, {"b-lyp", 4},
            {"b-p", 5}, {"pbe", 6}, {"pbe0", 7}, {"lc-wpbe", 8}
        };

        int commandCode = commandMap[functional_name];
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
                error->all(FLERR, "Functional name unknown");
                break;
        }
    } else {
        error->all(FLERR, "Unknown damping type");
    }

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
   Coeff: read from pair_coeff (Required)
          pair_coeff * * path_r0ab.csv path_c6ab.csv functional element1 element2 ...
------------------------------------------------------------------------- */

void PairD3::coeff(int narg, char **arg) {
    if (!allocated) allocate();

    int ntypes = atom->ntypes;
    if (narg != ntypes + 2) { error->all(FLERR, "Pair_coeff * * needs: element1 element2 ..."); }

    std::string element;
    int* atomic_numbers = (int*)malloc(sizeof(int)*ntypes);
    for (int i = 0; i < ntypes; i++) {
        element = arg[i+2];
        atomic_numbers[i] = find_atomic_number(element);
    }

    int count = 0;
    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            setflag[i][j] = 1;
            count++;
        }
    }

    if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

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
    }; // atomic <r^2>/<r^4> values

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

    free(atomic_numbers);
}

/* ----------------------------------------------------------------------
   Get derivative of C6 w.r.t. CN (used in PairD3::compute)

   C6 = C6(CN_A, CN_B) == W(CN_A, CN_B) / Z(CN_A, CN_B)

   This gives below from chain rule:
   d(C6)/dr = d(C6)/d(CN_A) * d(CN_A)/dr + d(C6)/d(CN_B) * d(CN_B)/dr

   So we can pre-calculate the d(C6)/d(CN_A), d(C6)/d(CN_B) part.

   d(C6)/d(CN_i) = (dW/d(CN_i) * Z - W * dZ/d(CN_i)) / (W * W)
        W : "denominator"
        Z : "numerator"
        dW/d(CN_i) : "d_denominator_i"
        dZ/d(CN_j) : "d_numerator_j"

    Z = Sum( L_ij(CN_A, CN_B) * C6_ref(CN_A_i, CN_B_j) ) over i, j
    W = Sum( L_ij(CN_A, CN_B) ) over i, j

   And the resulting derivative term is saved into
   "dc6_iji_tot", "dc6_ijj_tot" array,
   where we can find the value of d(C6)/d(CN_i)
   by knowing the index of "iat", and "jat". ("idx_linij")

   Also, c6 values will also be saved into "c6_ij_tot" array.

   Here, as we only interested in *pair* of atoms, assume "iat" >= "jat".
   Then "idx_linij" = "jat + (iat + 1) * iat / 2" have the order below.

     idx_linij | j = 0  j = 1  j = 2  j = 3    ...
---------------------------------------------
        i = 0  |     0
        i = 1  |     1      2
        i = 2  |     3      4      5
        i = 3  |     6      7      8      9
          ...  |    ...    ...    ...    ...   ...

------------------------------------------------------------------------- */

__global__ void kernel_get_dC6_dCNij(
    int maxij, float K3,
    double *cn, int *mxc, float *****c6ab, int *type,
    float *c6_ij_tot, float *dc6_iji_tot, float *dc6_ijj_tot
) {
    int iter = blockIdx.x * blockDim.x + threadIdx.x;

    if (iter < maxij) {
        int iat, jat;
        ij_at_linij(iter, iat, jat);

        const int atomtype_i = type[iat];
        const int atomtype_j = type[jat];

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

                    double expterm = exp(static_cast<double>(K3) * static_cast<double>(r)); // must be double
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
            const double denominator_rc = 1.0 / denominator; // must be double
            const double unit_frac = numerator * denominator_rc;
            c6_ij_tot[iter] = unit_frac;
            dc6_iji_tot[iter] = denominator_rc * fma(unit_frac, -d_denominator_i, d_numerator_i); // must be double
            dc6_ijj_tot[iter] = denominator_rc * fma(unit_frac, -d_denominator_j, d_numerator_j); // must be double
            //const double denominator_rc = 1.0 / denominator;
            //const float unit_frac = numerator * denominator_rc;
            //c6_ij_tot[iter] = unit_frac;
            //dc6_iji_tot[iter] = \
            static_cast<float>(d_numerator_i * denominator_rc) - static_cast<float>(d_denominator_i * denominator_rc) * unit_frac;
            //dc6_ijj_tot[iter] = \
            static_cast<float>(d_numerator_j * denominator_rc) - static_cast<float>(d_denominator_j * denominator_rc) * unit_frac;
        }
        else {
            c6_ij_tot[iter] = c6mem;
            dc6_iji_tot[iter] = 0.0f;
            dc6_ijj_tot[iter] = 0.0f;
        }
    }
}

void PairD3::get_dC6_dCNij() {
    int n = atom->natoms;
    int maxij = n * (n + 1) / 2;

    //START_CUDA_TIMER();

    int threadsPerBlock = 128;
    int blocksPerGrid = (maxij + threadsPerBlock - 1) / threadsPerBlock;
    kernel_get_dC6_dCNij<<<blocksPerGrid, threadsPerBlock>>>(
        maxij, K3,
        cn, mxc, c6ab, atomtype,
        c6_ij_tot, dc6_iji_tot, dc6_ijj_tot
    );
    cudaDeviceSynchronize();

    //STOP_CUDA_TIMER("get_dC6dCNij");
    //CHECK_CUDA_ERROR();
}

/* ----------------------------------------------------------------------
   Get lattice vectors (used in PairD3::compute)

   1) Save lattice vectors into "lat_v_1", "lat_v_2", "lat_v_3"
   2) Calculate repetition criteria for vdw, cn
   3) precaluclate tau (xyz shift due to cell repetition)

------------------------------------------------------------------------- */

void PairD3::set_lattice_vectors() {

    double boxxlo = domain->boxlo[0];
    double boxxhi = domain->boxhi[0];
    double boxylo = domain->boxlo[1];
    double boxyhi = domain->boxhi[1];
    double boxzlo = domain->boxlo[2];
    double boxzhi = domain->boxhi[2];
    double xy = domain->xy;
    double xz = domain->xz;
    double yz = domain->yz;

    lat_v_1[0] = (boxxhi - boxxlo) / AU_TO_ANG;
    lat_v_1[1] =               0.0;
    lat_v_1[2] =               0.0;
    lat_v_2[0] =                xy / AU_TO_ANG;
    lat_v_2[1] = (boxyhi - boxylo) / AU_TO_ANG;
    lat_v_2[2] =               0.0;
    lat_v_3[0] =                xz / AU_TO_ANG;
    lat_v_3[1] =                yz / AU_TO_ANG;
    lat_v_3[2] = (boxzhi - boxzlo) / AU_TO_ANG;

    int vdwrx_save = 2 * rep_vdw[0] + 1;
    int vdwry_save = 2 * rep_vdw[1] + 1;
    int vdwrz_save = 2 * rep_vdw[2] + 1;
    int cnrx_save = 2 * rep_cn[0] + 1;
    int cnry_save = 2 * rep_cn[1] + 1;
    int cnrz_save = 2 * rep_cn[2] + 1;

    set_lattice_repetition_criteria(rthr, rep_vdw);
    set_lattice_repetition_criteria(cn_thr, rep_cn);

    int vdw_range_x = 2 * rep_vdw[0] + 1;
    int vdw_range_y = 2 * rep_vdw[1] + 1;
    int vdw_range_z = 2 * rep_vdw[2] + 1;
    int tau_loop_size_vdw = vdw_range_x * vdw_range_y * vdw_range_z * 3;
    if (tau_loop_size_vdw != tau_idx_vdw_total_size) {
        if (tau_idx_vdw != nullptr) {
            for (int i = 0; i < vdwrx_save; i++) {
                for (int j = 0; j < vdwry_save; j++) {
                    for (int k = 0; k < vdwrz_save; k++) {
                        cudaFree(tau_vdw[i][j][k]);
                    }
                    cudaFree(tau_vdw[i][j]);
                }
                cudaFree(tau_vdw[i]);
            }
            cudaFree(tau_vdw);
            cudaFree(tau_idx_vdw);
        }
        tau_idx_vdw_total_size = tau_loop_size_vdw;
        cudaMallocManaged(&tau_vdw, vdw_range_x * sizeof(float***));
        for (int i = 0; i < vdw_range_x; i++) {
            cudaMallocManaged(&tau_vdw[i], vdw_range_y * sizeof(float**));
            for (int j = 0; j < vdw_range_y; j++) {
                cudaMallocManaged(&tau_vdw[i][j], vdw_range_z * sizeof(float*));
                for (int k = 0; k < vdw_range_z; k++) {
                    cudaMallocManaged(&tau_vdw[i][j][k], 3 * sizeof(float));
                }
            }
        }
        cudaMallocManaged(&tau_idx_vdw, tau_idx_vdw_total_size * sizeof(int));
    }

    int cn_range_x  = 2 * rep_cn[0] + 1;
    int cn_range_y  = 2 * rep_cn[1] + 1;
    int cn_range_z  = 2 * rep_cn[2] + 1;
    int tau_loop_size_cn = cn_range_x * cn_range_y * cn_range_z * 3;
    if (tau_loop_size_cn != tau_idx_cn_total_size) {
        if (tau_idx_cn != nullptr) {
            for (int i = 0; i < cnrx_save; i++) {
                for (int j = 0; j < cnry_save; j++) {
                    for (int k = 0; k < cnrz_save; k++) {
                        cudaFree(tau_cn[i][j][k]);
                    }
                    cudaFree(tau_cn[i][j]);
                }
                cudaFree(tau_cn[i]);
            }
            cudaFree(tau_cn);
            cudaFree(tau_idx_cn);
        }
        tau_idx_cn_total_size = tau_loop_size_cn;
        cudaMallocManaged(&tau_cn, cn_range_x * sizeof(float***));
        for (int i = 0; i < cn_range_x; i++) {
            cudaMallocManaged(&tau_cn[i], cn_range_y * sizeof(float**));
            for (int j = 0; j < cn_range_y; j++) {
                cudaMallocManaged(&tau_cn[i][j], cn_range_z * sizeof(float*));
                for (int k = 0; k < cn_range_z; k++) {
                    cudaMallocManaged(&tau_cn[i][j][k], 3 * sizeof(float));
                }
            }
        }
        cudaMallocManaged(&tau_idx_cn, tau_idx_cn_total_size * sizeof(int));
    }

    //CHECK_CUDA_ERROR();
}

/* ----------------------------------------------------------------------
   Set repetition criteria (used in PairD3::compute)

   Needed as Periodic Boundary Condition should be considered.

   As the cell may *not* be orthorhombic,
   the dot product should be used between x/y/z direction and
   corresponding cross product vector.
------------------------------------------------------------------------- */

void PairD3::set_lattice_repetition_criteria(float r_threshold, int* rep_v) {
    double r_cutoff = sqrt(r_threshold);
    double lat_cp_12[3], lat_cp_23[3], lat_cp_31[3];
    double cos_value;

    MathExtra::cross3(lat_v_1, lat_v_2, lat_cp_12);
    MathExtra::cross3(lat_v_2, lat_v_3, lat_cp_23);
    MathExtra::cross3(lat_v_3, lat_v_1, lat_cp_31);

    cos_value = MathExtra::dot3(lat_cp_23, lat_v_1) / MathExtra::len3(lat_cp_23);
    rep_v[0] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
    cos_value = MathExtra::dot3(lat_cp_31, lat_v_2) / MathExtra::len3(lat_cp_31);
    rep_v[1] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;
    cos_value = MathExtra::dot3(lat_cp_12, lat_v_3) / MathExtra::len3(lat_cp_12);
    rep_v[2] = static_cast<int>(std::abs(r_cutoff / cos_value)) + 1;

    if (domain->xperiodic == 0) { rep_v[0] = 0; }
    if (domain->yperiodic == 0) { rep_v[1] = 0; }
    if (domain->zperiodic == 0) { rep_v[2] = 0; }
}

/* ----------------------------------------------------------------------
   Calculate Coordination Number (used in PairD3::compute)
------------------------------------------------------------------------- */

__global__ void kernel_get_coordination_number(
    int maxij, int maxtau, float cn_thr, float K1,
    float *rcov, int *rep_cn, float ****tau_cn, int *tau_idx_cn, int *type, float **x,
    double *cn
) {
    int iter = blockIdx.x * blockDim.x + threadIdx.x;

    if (iter < maxij) {
        int iat, jat;
        ij_at_linij(iter, iat, jat);

        float cn_local = 0.0f;

        if (iat == jat) {
            const float rcov_sum = rcov[type[iat]] * 2.0f;
            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_cn[k-2];
                const int idx2 = tau_idx_cn[k-1];
                const int idx3 = tau_idx_cn[k];
                if (idx1 == rep_cn[0] && idx2 == rep_cn[1] && idx3 == rep_cn[2]) { continue; }
                const float rx = tau_cn[idx1][idx2][idx3][0];
                const float ry = tau_cn[idx1][idx2][idx3][1];
                const float rz = tau_cn[idx1][idx2][idx3][2];
                const float r2 = rx * rx + ry * ry + rz * rz;
                if (r2 <= cn_thr) {
                    const float r_rc = rsqrtf(r2);
                    const float damp = 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
                    cn_local += damp;
                }
            }
            atomicAdd(&cn[iat], cn_local);
        }

        else {
            const float rcov_sum = rcov[type[iat]] + rcov[type[jat]];
            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_cn[k-2];
                const int idx2 = tau_idx_cn[k-1];
                const int idx3 = tau_idx_cn[k];
                const float rx = x[jat][0] - x[iat][0] + tau_cn[idx1][idx2][idx3][0];
                const float ry = x[jat][1] - x[iat][1] + tau_cn[idx1][idx2][idx3][1];
                const float rz = x[jat][2] - x[iat][2] + tau_cn[idx1][idx2][idx3][2];
                const float r2 = rx * rx + ry * ry + rz * rz;
                if (r2 <= cn_thr) {
                    const float r_rc = rsqrtf(r2);
                    const float damp = 1.0f / (1.0f + expf(-K1 * ((rcov_sum * r_rc) - 1.0f)));
                    cn_local += damp;
                }
            }
            atomicAdd(&cn[iat], cn_local);
            atomicAdd(&cn[jat], cn_local);
        }
    }
}

void PairD3::get_coordination_number() {
    int n = atom->natoms;
    int maxij = n * (n + 1) / 2;
    int maxtau = tau_idx_cn_total_size;

    for (int i = 0; i < n; i++) {
        cn[i] = 0.0;
    }

    //START_CUDA_TIMER();

    int threadsPerBlock = 128;
    int blocksPerGrid = (maxij + threadsPerBlock - 1) / threadsPerBlock;
    kernel_get_coordination_number<<<blocksPerGrid, threadsPerBlock>>>(
        maxij, maxtau, cn_thr, K1,
        rcov, rep_cn, tau_cn, tau_idx_cn, atomtype, x,
        cn
    );
    cudaDeviceSynchronize();

    //STOP_CUDA_TIMER("get_coord");
    //CHECK_CUDA_ERROR();

    get_dC6_dCNij();
}

/* ----------------------------------------------------------------------
   reallcate memory if the number of atoms has changed (used in PairD3::compute)
------------------------------------------------------------------------- */

void PairD3::reallocate_arrays() {

    /* -------------- Destroy previous arrays -------------- */
    cudaFree(cn);
    for (int i = 0; i < n_save; i++) { cudaFree(x[i]); }; cudaFree(x);
    cudaFree(dc6i);
    for (int i = 0; i < n_save; i++) { cudaFree(f[i]); }; cudaFree(f);

    cudaFree(dc6_iji_tot);
    cudaFree(dc6_ijj_tot);
    cudaFree(c6_ij_tot);

    cudaFree(atomtype);

    /* -------------- Destroy previous arrays -------------- */

    /* -------------- Create new arrays -------------- */
    int n = atom->natoms;
    n_save = n;

    cudaMallocManaged(&cn, n * sizeof(double));
    cudaMallocManaged(&x, n * sizeof(float*)); for (int i = 0; i < n; i++) { cudaMallocManaged(&x[i], 3 * sizeof(float)); }
    cudaMallocManaged(&dc6i, n * sizeof(double));
    cudaMallocManaged(&f, n * sizeof(double*)); for (int i = 0; i < n; i++) { cudaMallocManaged(&f[i], 3 * sizeof(double)); }

    set_lattice_vectors();

    int n_ij_combination = n * (n + 1) / 2;
    cudaMallocManaged(&dc6_iji_tot, n_ij_combination * sizeof(float));
    cudaMallocManaged(&dc6_ijj_tot, n_ij_combination * sizeof(float));
    cudaMallocManaged(&c6_ij_tot,   n_ij_combination * sizeof(float));

    cudaMallocManaged(&atomtype, n * sizeof(int));

    /* -------------- Create new arrays -------------- */

    //CHECK_CUDA_ERROR();
}

/* ----------------------------------------------------------------------
  Initialize atomic positions & types (used in PairD3::compute)

   As the default xyz from lammps does not assure that atoms are within unit cell,
   this function shifts atoms into the unit cell.
------------------------------------------------------------------------- */

void PairD3::load_atom_info() {
    double lat[3][3];
    lat[0][0] = lat_v_1[0];
    lat[0][1] = lat_v_2[0];
    lat[0][2] = lat_v_3[0];
    lat[1][0] = lat_v_1[1];
    lat[1][1] = lat_v_2[1];
    lat[1][2] = lat_v_3[1];
    lat[2][0] = lat_v_1[2];
    lat[2][1] = lat_v_2[2];
    lat[2][2] = lat_v_3[2];

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

    double a[3] = { 0.0 };
    for (int iat = 0; iat < atom->natoms; iat++) {
        for (int i = 0; i < 3; i++) {
            a[i] = lat_inv[i][0] * (atom->x)[iat][0] + lat_inv[i][1] * (atom->x)[iat][1] + lat_inv[i][2] * (atom->x)[iat][2];
            if      (a[i] > 1) { while (a[i] > 1) { a[i]--; } }
            else if (a[i] < 0) { while (a[i] < 0) { a[i]++; } }
        }

        for (int i = 0; i < 3; i++) {
            x[iat][i] = (lat[i][0] * a[0] + lat[i][1] * a[1] + lat[i][2] * a[2]) / AU_TO_ANG;
        }
    }
}

/* ----------------------------------------------------------------------
   Precalculate tau array
------------------------------------------------------------------------- */

void PairD3::precalculate_tau_array() {
    int xlim = rep_vdw[0];
    int ylim = rep_vdw[1];
    int zlim = rep_vdw[2];

    int index = 0;
    for (int taux = -xlim; taux <= xlim; taux++) {
        for (int tauy = -ylim; tauy <= ylim; tauy++) {
            for (int tauz = -zlim; tauz <= zlim; tauz++) {
                tau_vdw[taux + xlim][tauy + ylim][tauz + zlim][0] = lat_v_1[0] * taux + lat_v_2[0] * tauy + lat_v_3[0] * tauz;
                tau_vdw[taux + xlim][tauy + ylim][tauz + zlim][1] = lat_v_1[1] * taux + lat_v_2[1] * tauy + lat_v_3[1] * tauz;
                tau_vdw[taux + xlim][tauy + ylim][tauz + zlim][2] = lat_v_1[2] * taux + lat_v_2[2] * tauy + lat_v_3[2] * tauz;
                tau_idx_vdw[index++] = taux + xlim;
                tau_idx_vdw[index++] = tauy + ylim;
                tau_idx_vdw[index++] = tauz + zlim;
            }
        }
    }

    xlim = rep_cn[0];
    ylim = rep_cn[1];
    zlim = rep_cn[2];

    index = 0;
    for (int taux = -xlim; taux <= xlim; taux++) {
        for (int tauy = -ylim; tauy <= ylim; tauy++) {
            for (int tauz = -zlim; tauz <= zlim; tauz++) {
                tau_cn[taux + xlim][tauy + ylim][tauz + zlim][0] = lat_v_1[0] * taux + lat_v_2[0] * tauy + lat_v_3[0] * tauz;
                tau_cn[taux + xlim][tauy + ylim][tauz + zlim][1] = lat_v_1[1] * taux + lat_v_2[1] * tauy + lat_v_3[1] * tauz;
                tau_cn[taux + xlim][tauy + ylim][tauz + zlim][2] = lat_v_1[2] * taux + lat_v_2[2] * tauy + lat_v_3[2] * tauz;
                tau_idx_cn[index++] = taux + xlim;
                tau_idx_cn[index++] = tauy + ylim;
                tau_idx_cn[index++] = tauz + zlim;
            }
        }
    }
}


/* ----------------------------------------------------------------------
   Get forces (Zero damping)
------------------------------------------------------------------------- */

__global__ void kernel_get_forces_without_dC6_zero_damping(
    int maxij, int maxtau, float rthr, float s6, float s8, float a1, float a2, float alp6, float alp8,
    float *r2r4, float **r0ab, int *rep_vdw, float ****tau_vdw, int *tau_idx_vdw, int *type, float **x,
    float *c6_ij_tot, float *dc6_iji_tot, float *dc6_ijj_tot,
    double *dc6i, double *disp, double **f, double **sigma
) {
    int iter = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sigma_00[128];
    __shared__ float sigma_01[128];
    __shared__ float sigma_02[128];
    __shared__ float sigma_10[128];
    __shared__ float sigma_11[128];
    __shared__ float sigma_12[128];
    __shared__ float sigma_20[128];
    __shared__ float sigma_21[128];
    __shared__ float sigma_22[128];
    __shared__ float disp_shared[128];

    float sigma_local_00 = 0.0f;
    float sigma_local_01 = 0.0f;
    float sigma_local_02 = 0.0f;
    float sigma_local_10 = 0.0f;
    float sigma_local_11 = 0.0f;
    float sigma_local_12 = 0.0f;
    float sigma_local_20 = 0.0f;
    float sigma_local_21 = 0.0f;
    float sigma_local_22 = 0.0f;
    float disp_local = 0.0f;

    if (iter < maxij) {
        int iat, jat;
        ij_at_linij(iter, iat, jat);

        float f_local[3] = { 0.0f };
        float dc6i_local_i = 0.0f;
        float dc6i_local_j = 0.0f;

        const float c6 = c6_ij_tot[iter];
        const float dc6iji = dc6_iji_tot[iter];
        const float dc6ijj = dc6_ijj_tot[iter];

        if (iat == jat) {
            const int atomtype_i = type[iat];
            const float r0 = r0ab[atomtype_i][atomtype_i];
            const float unit_r2r4 = r2r4[atomtype_i];
            const float r42 = unit_r2r4 * unit_r2r4;
            const float unit_a1 = (a1 * r0);
            const float unit_a2 = (a2 * r0);
            const float s8r42 = s8 * r42;

            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_vdw[k-2];
                const int idx2 = tau_idx_vdw[k-1];
                const int idx3 = tau_idx_vdw[k];

                if (idx1 == rep_vdw[0] && idx2 == rep_vdw[1] && idx3 == rep_vdw[2]) { continue; }
                const float rij[3] = {
                    tau_vdw[idx1][idx2][idx3][0],
                    tau_vdw[idx1][idx2][idx3][1],
                    tau_vdw[idx1][idx2][idx3][2]
                };
                const float r2 = lensq3(rij);
                if (r2 > rthr) { continue; }

                const float r_rc = rsqrtf(r2);
                float unit_rc_a1 = unit_a1 * r_rc;
                float t6 = unit_rc_a1 * unit_rc_a1; // ^2
                t6 *= unit_rc_a1; // ^3
                t6 *= t6; // ^6
                t6 *= unit_rc_a1; // ^7
                t6 *= t6; // ^14
                const float damp6 = 1.0f / fmaf(t6, 6.0f, 1.0f);
                float unit_rc_a2 = unit_a2 * r_rc;
                float t8 = unit_rc_a2 * unit_rc_a2; // ^2
                t8 *= t8; // ^4
                t8 *= t8; // ^8
                t8 *= t8; // ^16
                const float damp8 = 1.0f / fmaf(t8, 6.0f, 1.0f);
                const float r2_rc = r_rc * r_rc; // 1.0 / r2
                const float r6_rc = r2_rc * r2_rc * r2_rc;
                const float r8_rc = r6_rc * r2_rc;
                const float x1 = 3.0f * c6 * r8_rc * fmaf(r2_rc, s8r42 * damp8 * fmaf(3.0f * alp8 * t8, damp8, -4.0f), s6 * damp6 * fmaf(alp6 * t6, damp6, -1.0f));
                //const float x1 = 0.5 * 6.0 * c6 * r8_rc * (s6 * damp6 * (14.0 * t6 * damp6 - 1.0) + s8r42 * r2_rc * damp8 * (48.0 * t8 * damp8 - 4.0));
                //3.0 * alp6 = 48.0

                const float vec[3] = {
                    x1 * rij[0],
                    x1 * rij[1],
                    x1 * rij[2]
                };

                sigma_local_00 += vec[0] * rij[0];
                sigma_local_01 += vec[0] * rij[1];
                sigma_local_02 += vec[0] * rij[2];
                sigma_local_10 += vec[1] * rij[0];
                sigma_local_11 += vec[1] * rij[1];
                sigma_local_12 += vec[1] * rij[2];
                sigma_local_20 += vec[2] * rij[0];
                sigma_local_21 += vec[2] * rij[1];
                sigma_local_22 += vec[2] * rij[2];

                const float dc6_rest = 0.5f * r6_rc * fmaf(3.0f * r2_rc, s8r42 * damp8, s6 * damp6);
                //const float dc6_rest = 0.5 * r6_rc * (s6 * damp6 + 3.0 * s8r42 * damp8 * r2_rc);
                disp_local -= dc6_rest * c6;
                dc6i_local_i += dc6_rest * dc6iji;
                dc6i_local_j += dc6_rest * dc6ijj;
            }
            atomicAdd(&dc6i[iat], dc6i_local_i);
            atomicAdd(&dc6i[jat], dc6i_local_j);
        }

        else {
            const int atomtype_i = type[iat];
            const int atomtype_j = type[jat];
            const float r0 = r0ab[atomtype_i][atomtype_j];
            const float r42 = r2r4[atomtype_i] * r2r4[atomtype_j];
            const float unit_a1 = (a1 * r0);
            const float unit_a2 = (a2 * r0);
            const float s8r42 = s8 * r42;

            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_vdw[k-2];
                const int idx2 = tau_idx_vdw[k-1];
                const int idx3 = tau_idx_vdw[k];

                const float rij[3] = {
                    x[jat][0] - x[iat][0] + tau_vdw[idx1][idx2][idx3][0],
                    x[jat][1] - x[iat][1] + tau_vdw[idx1][idx2][idx3][1],
                    x[jat][2] - x[iat][2] + tau_vdw[idx1][idx2][idx3][2]
                };
                const float r2 = lensq3(rij);
                if (r2 > rthr) { continue; }

                const float r_rc = rsqrtf(r2);
                float unit_rc_a1 = unit_a1 * r_rc;
                float t6 = unit_rc_a1 * unit_rc_a1; // ^2
                t6 *= unit_rc_a1; // ^3
                t6 *= t6; // ^6
                t6 *= unit_rc_a1; // ^7
                t6 *= t6; // ^14
                const float damp6 = 1.0f / fmaf(t6, 6.0f, 1.0f);
                float unit_rc_a2 = unit_a2 * r_rc;
                float t8 = unit_rc_a2 * unit_rc_a2; // ^2
                t8 *= t8; // ^4
                t8 *= t8; // ^8
                t8 *= t8; // ^16
                const float damp8 = 1.0f / fmaf(t8, 6.0f, 1.0f);
                const float r2_rc = r_rc * r_rc; // 1.0 / r2
                const float r6_rc = r2_rc * r2_rc * r2_rc;
                const float r8_rc = r6_rc * r2_rc;
                const float x1 = 6.0f * c6 * r8_rc * fmaf(r2_rc, s8r42 * damp8 * fmaf(3.0f * alp8 * t8, damp8, -4.0f), s6 * damp6 * fmaf(alp6 * t6, damp6, -1.0f));
                //const float x1 = 6.0 * c6 * r8_rc * (s6 * damp6 * (14.0 * t6 * damp6 - 1.0) + s8r42 * r2_rc * damp8 * (48.0 * t8 * damp8 - 4.0));
                //3.0 * alp6 = 48.0

                const float vec[3] = {
                    x1 * rij[0],
                    x1 * rij[1],
                    x1 * rij[2]
                };

                f_local[0] -= vec[0];
                f_local[1] -= vec[1];
                f_local[2] -= vec[2];

                sigma_local_00 += vec[0] * rij[0];
                sigma_local_01 += vec[0] * rij[1];
                sigma_local_02 += vec[0] * rij[2];
                sigma_local_10 += vec[1] * rij[0];
                sigma_local_11 += vec[1] * rij[1];
                sigma_local_12 += vec[1] * rij[2];
                sigma_local_20 += vec[2] * rij[0];
                sigma_local_21 += vec[2] * rij[1];
                sigma_local_22 += vec[2] * rij[2];

                const float dc6_rest = r6_rc * fmaf(3.0f * r2_rc, s8r42 * damp8, s6 * damp6);
                //const float dc6_rest = r6_rc * (s6 * damp6 + 3.0 * s8r42 * damp8 * r2_rc);
                disp_local -= dc6_rest * c6;
                dc6i_local_i += dc6_rest * dc6iji;
                dc6i_local_j += dc6_rest * dc6ijj;
            }
            atomicAdd(&dc6i[iat], dc6i_local_i);
            atomicAdd(&dc6i[jat], dc6i_local_j);
            atomicAdd(&f[iat][0], f_local[0]);
            atomicAdd(&f[iat][1], f_local[1]);
            atomicAdd(&f[iat][2], f_local[2]);
            atomicAdd(&f[jat][0], -f_local[0]);
            atomicAdd(&f[jat][1], -f_local[1]);
            atomicAdd(&f[jat][2], -f_local[2]);
        }
    }

    sigma_00[threadIdx.x] = sigma_local_00;
    sigma_01[threadIdx.x] = sigma_local_01;
    sigma_02[threadIdx.x] = sigma_local_02;
    sigma_10[threadIdx.x] = sigma_local_10;
    sigma_11[threadIdx.x] = sigma_local_11;
    sigma_12[threadIdx.x] = sigma_local_12;
    sigma_20[threadIdx.x] = sigma_local_20;
    sigma_21[threadIdx.x] = sigma_local_21;
    sigma_22[threadIdx.x] = sigma_local_22;
    disp_shared[threadIdx.x] = disp_local;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sigma_00[threadIdx.x] += sigma_00[threadIdx.x + s];
            sigma_01[threadIdx.x] += sigma_01[threadIdx.x + s];
            sigma_02[threadIdx.x] += sigma_02[threadIdx.x + s];
            sigma_10[threadIdx.x] += sigma_10[threadIdx.x + s];
            sigma_11[threadIdx.x] += sigma_11[threadIdx.x + s];
            sigma_12[threadIdx.x] += sigma_12[threadIdx.x + s];
            sigma_20[threadIdx.x] += sigma_20[threadIdx.x + s];
            sigma_21[threadIdx.x] += sigma_21[threadIdx.x + s];
            sigma_22[threadIdx.x] += sigma_22[threadIdx.x + s];
            disp_shared[threadIdx.x] += disp_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&sigma[0][0], sigma_00[0]);
        atomicAdd(&sigma[0][1], sigma_01[0]);
        atomicAdd(&sigma[0][2], sigma_02[0]);
        atomicAdd(&sigma[1][0], sigma_10[0]);
        atomicAdd(&sigma[1][1], sigma_11[0]);
        atomicAdd(&sigma[1][2], sigma_12[0]);
        atomicAdd(&sigma[2][0], sigma_20[0]);
        atomicAdd(&sigma[2][1], sigma_21[0]);
        atomicAdd(&sigma[2][2], sigma_22[0]);
        atomicAdd(disp, disp_shared[0]);
    }
}

void PairD3::get_forces_without_dC6_zero_damping() {
    int n = atom->natoms;
    int maxij = n * (n + 1) / 2;
    int maxtau = tau_idx_vdw_total_size;

    *dispall = 0.0;

    for (int dim = 0; dim < n; dim++) { dc6i[dim] = 0.0; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f[i][j] = 0.0;
        }
    }

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            sigma[ii][jj] = 0.0;
        }
    }

    //START_CUDA_TIMER();

    int threadsPerBlock = 128;
    int blocksPerGrid = (maxij + threadsPerBlock - 1) / threadsPerBlock;
    kernel_get_forces_without_dC6_zero_damping<<<blocksPerGrid, threadsPerBlock>>>(
        maxij, maxtau, rthr, s6, s8, a1, a2, alp6, alp8,
        r2r4, r0ab, rep_vdw, tau_vdw, tau_idx_vdw, atomtype, x,
        c6_ij_tot, dc6_iji_tot, dc6_ijj_tot,
        dc6i, dispall, f, sigma
    );
    cudaDeviceSynchronize();
    disp_total = *dispall;

    //STOP_CUDA_TIMER("get_forces_without");
    //CHECK_CUDA_ERROR();
}

/* ----------------------------------------------------------------------
   Get forces (Zero damping)
------------------------------------------------------------------------- */

// Not implemented yet
void PairD3::get_forces_without_dC6_zero_damping_modified() {

}

/* ----------------------------------------------------------------------
   Get forces (BJ damping)
------------------------------------------------------------------------- */

__global__ void kernel_get_forces_without_dC6_bj_damping(
    int maxij, int maxtau, float rthr, float s6, float s8, float a1, float a2,
    float *r2r4, int *rep_vdw, float ****tau_vdw, int *tau_idx_vdw, int *type, float **x,
    float *c6_ij_tot, float *dc6_iji_tot, float *dc6_ijj_tot,
    double *dc6i, double *disp, double **f, double **sigma
) {
    int iter = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sigma_00[128];
    __shared__ float sigma_01[128];
    __shared__ float sigma_02[128];
    __shared__ float sigma_10[128];
    __shared__ float sigma_11[128];
    __shared__ float sigma_12[128];
    __shared__ float sigma_20[128];
    __shared__ float sigma_21[128];
    __shared__ float sigma_22[128];
    __shared__ float disp_shared[128];

    float sigma_local_00 = 0.0f;
    float sigma_local_01 = 0.0f;
    float sigma_local_02 = 0.0f;
    float sigma_local_10 = 0.0f;
    float sigma_local_11 = 0.0f;
    float sigma_local_12 = 0.0f;
    float sigma_local_20 = 0.0f;
    float sigma_local_21 = 0.0f;
    float sigma_local_22 = 0.0f;
    float disp_local = 0.0f;

    if (iter < maxij) {
        int iat, jat;
        ij_at_linij(iter, iat, jat);

        float f_local[3] = { 0.0f };
        float dc6i_local_i = 0.0f;
        float dc6i_local_j = 0.0f;

        const float c6 = c6_ij_tot[iter];
        const float dc6iji = dc6_iji_tot[iter];
        const float dc6ijj = dc6_ijj_tot[iter];

        if (iat == jat) {
            const float unit_r2r4 = r2r4[type[iat]];
            const float r42x3 = unit_r2r4 * unit_r2r4 * 3.0f;
            const float R0 = fmaf(a1, sqrtf(r42x3), a2);
            const float R0_2 = R0 * R0;
            const float R0_6 = R0_2 * R0_2 * R0_2;
            const float R0_8 = R0_6 * R0_2;
            const float s8r42x3 = s8 * r42x3;

            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_vdw[k-2];
                const int idx2 = tau_idx_vdw[k-1];
                const int idx3 = tau_idx_vdw[k];

                if (idx1 == rep_vdw[0] && idx2 == rep_vdw[1] && idx3 == rep_vdw[2]) { continue; }
                const float rij[3] = {
                    tau_vdw[idx1][idx2][idx3][0],
                    tau_vdw[idx1][idx2][idx3][1],
                    tau_vdw[idx1][idx2][idx3][2]
                };
                const float r2 = lensq3(rij);
                if (r2 > rthr) { continue; }

                const float r = sqrtf(r2);
                const float r5 = r2 * r2 * r;
                const float r7 = r5 * r2;
                const float t6_rc = 1.0f / fmaf(r5, r, R0_6);
                const float t8_rc = 1.0f / fmaf(r7, r, R0_8);
                const float t6_sqrc = t6_rc * t6_rc;
                const float t8_sqrc = t8_rc * t8_rc;
                const float x1 = -c6 * fmaf(4.0f * s8r42x3 * r7, t8_sqrc, 3.0f * s6 * r5 * t6_sqrc);
                //const float x1 = 0.5 * -c6 * (6.0 * s6 * r5 * t6_sqrc + 8.0 * s8r42x3 * r7 * t8_sqrc;

                const float r_rc = 1.0f / r; // rsqrt(r2)
                const float vec[3] = {
                    x1 * rij[0] * r_rc,
                    x1 * rij[1] * r_rc,
                    x1 * rij[2] * r_rc
                };

                sigma_local_00 += vec[0] * rij[0];
                sigma_local_01 += vec[0] * rij[1];
                sigma_local_02 += vec[0] * rij[2];
                sigma_local_10 += vec[1] * rij[0];
                sigma_local_11 += vec[1] * rij[1];
                sigma_local_12 += vec[1] * rij[2];
                sigma_local_20 += vec[2] * rij[0];
                sigma_local_21 += vec[2] * rij[1];
                sigma_local_22 += vec[2] * rij[2];

                const float dc6_rest = 0.5f * fmaf(s8r42x3, t8_rc, s6 * t6_rc);
                //const float dc6_rest = 0.5 * s6 * t6_rc + s8r42x3 * t8_rc;
                disp_local -= dc6_rest * c6;
                dc6i_local_i += dc6_rest * dc6iji;
                dc6i_local_j += dc6_rest * dc6ijj;
            }
            atomicAdd(&dc6i[iat], dc6i_local_i);
            atomicAdd(&dc6i[jat], dc6i_local_j);
        }

        else {
            const float r42x3 = r2r4[type[iat]] * r2r4[type[jat]] * 3.0f;
            const float R0 = fmaf(a1, sqrtf(r42x3), a2);
            const float R0_2 = R0 * R0;
            const float R0_6 = R0_2 * R0_2 * R0_2;
            const float R0_8 = R0_6 * R0_2;
            const float s8r42x3 = s8 * r42x3;

            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_vdw[k-2];
                const int idx2 = tau_idx_vdw[k-1];
                const int idx3 = tau_idx_vdw[k];
                const float rij[3] = {
                    x[jat][0] - x[iat][0] + tau_vdw[idx1][idx2][idx3][0],
                    x[jat][1] - x[iat][1] + tau_vdw[idx1][idx2][idx3][1],
                    x[jat][2] - x[iat][2] + tau_vdw[idx1][idx2][idx3][2]
                };
                const float r2 = lensq3(rij);
                if (r2 > rthr) { continue; }

                const float r = sqrtf(r2);
                const float r5 = r2 * r2 * r;
                const float r7 = r5 * r2;
                const float t6_rc = 1.0f / fmaf(r5, r, R0_6);
                const float t8_rc = 1.0f / fmaf(r7, r, R0_8);
                const float t6_sqrc = t6_rc * t6_rc;
                const float t8_sqrc = t8_rc * t8_rc;
                const float x1 = -c6 * fmaf(8.0f * s8r42x3 * r7, t8_sqrc, 6.0f * s6 * r5 * t6_sqrc);
                //const float x1 = -c6 * (6.0 * s6 * r5 * t6_sqrc + 8.0 * s8r42x3 * r7 * t8_sqrc;

                const float r_rc = 1.0f / r; // rsqrt(r2)
                const float vec[3] = {
                    x1 * rij[0] * r_rc,
                    x1 * rij[1] * r_rc,
                    x1 * rij[2] * r_rc
                };

                f_local[0] -= vec[0];
                f_local[1] -= vec[1];
                f_local[2] -= vec[2];

                sigma_local_00 += vec[0] * rij[0];
                sigma_local_01 += vec[0] * rij[1];
                sigma_local_02 += vec[0] * rij[2];
                sigma_local_10 += vec[1] * rij[0];
                sigma_local_11 += vec[1] * rij[1];
                sigma_local_12 += vec[1] * rij[2];
                sigma_local_20 += vec[2] * rij[0];
                sigma_local_21 += vec[2] * rij[1];
                sigma_local_22 += vec[2] * rij[2];

                const float dc6_rest = fmaf(s8r42x3, t8_rc, s6 * t6_rc);
                //const float dc6_rest = s6 * t6_rc + s8r42x3 * t8_rc;
                disp_local -= dc6_rest * c6;
                dc6i_local_i += dc6_rest * dc6iji;
                dc6i_local_j += dc6_rest * dc6ijj;
            }
            atomicAdd(&dc6i[iat], dc6i_local_i);
            atomicAdd(&dc6i[jat], dc6i_local_j);
            atomicAdd(&f[iat][0], f_local[0]);
            atomicAdd(&f[iat][1], f_local[1]);
            atomicAdd(&f[iat][2], f_local[2]);
            atomicAdd(&f[jat][0], -f_local[0]);
            atomicAdd(&f[jat][1], -f_local[1]);
            atomicAdd(&f[jat][2], -f_local[2]);
        }
    }

    sigma_00[threadIdx.x] = sigma_local_00;
    sigma_01[threadIdx.x] = sigma_local_01;
    sigma_02[threadIdx.x] = sigma_local_02;
    sigma_10[threadIdx.x] = sigma_local_10;
    sigma_11[threadIdx.x] = sigma_local_11;
    sigma_12[threadIdx.x] = sigma_local_12;
    sigma_20[threadIdx.x] = sigma_local_20;
    sigma_21[threadIdx.x] = sigma_local_21;
    sigma_22[threadIdx.x] = sigma_local_22;
    disp_shared[threadIdx.x] = disp_local;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sigma_00[threadIdx.x] += sigma_00[threadIdx.x + s];
            sigma_01[threadIdx.x] += sigma_01[threadIdx.x + s];
            sigma_02[threadIdx.x] += sigma_02[threadIdx.x + s];
            sigma_10[threadIdx.x] += sigma_10[threadIdx.x + s];
            sigma_11[threadIdx.x] += sigma_11[threadIdx.x + s];
            sigma_12[threadIdx.x] += sigma_12[threadIdx.x + s];
            sigma_20[threadIdx.x] += sigma_20[threadIdx.x + s];
            sigma_21[threadIdx.x] += sigma_21[threadIdx.x + s];
            sigma_22[threadIdx.x] += sigma_22[threadIdx.x + s];
            disp_shared[threadIdx.x] += disp_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&sigma[0][0], sigma_00[0]);
        atomicAdd(&sigma[0][1], sigma_01[0]);
        atomicAdd(&sigma[0][2], sigma_02[0]);
        atomicAdd(&sigma[1][0], sigma_10[0]);
        atomicAdd(&sigma[1][1], sigma_11[0]);
        atomicAdd(&sigma[1][2], sigma_12[0]);
        atomicAdd(&sigma[2][0], sigma_20[0]);
        atomicAdd(&sigma[2][1], sigma_21[0]);
        atomicAdd(&sigma[2][2], sigma_22[0]);
        atomicAdd(disp, disp_shared[0]);
    }
}

void PairD3::get_forces_without_dC6_bj_damping() {
    int n = atom->natoms;
    int maxij = n * (n + 1) / 2;
    int maxtau = tau_idx_vdw_total_size;

    *dispall = 0.0;

    for (int dim = 0; dim < n; dim++) { dc6i[dim] = 0.0; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f[i][j] = 0.0;
        }
    }

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            sigma[ii][jj] = 0.0;
        }
    }

    //START_CUDA_TIMER();

    int threadsPerBlock = 128;
    int blocksPerGrid = (maxij + threadsPerBlock - 1) / threadsPerBlock;
    kernel_get_forces_without_dC6_bj_damping<<<blocksPerGrid, threadsPerBlock>>>(
        maxij, maxtau, rthr, s6, s8, a1, a2,
        r2r4, rep_vdw, tau_vdw, tau_idx_vdw, atomtype, x,
        c6_ij_tot, dc6_iji_tot, dc6_ijj_tot,
        dc6i, dispall, f, sigma
    );
    cudaDeviceSynchronize();
    disp_total = *dispall;

    //STOP_CUDA_TIMER("get_forces_without");
    //CHECK_CUDA_ERROR();
}

/* ----------------------------------------------------------------------
   Get forces
------------------------------------------------------------------------- */

__global__ void kernel_get_forces_with_dC6(
    int maxij, int maxtau, float cn_thr, float K1,
    double *dc6i, float *rcov, int *rep_cn, float ****tau_cn, int *tau_idx_cn, int *type, float **x,
    double **f, double **sigma
) {
    int iter = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sigma_00[128];
    __shared__ float sigma_01[128];
    __shared__ float sigma_02[128];
    __shared__ float sigma_10[128];
    __shared__ float sigma_11[128];
    __shared__ float sigma_12[128];
    __shared__ float sigma_20[128];
    __shared__ float sigma_21[128];
    __shared__ float sigma_22[128];

    float sigma_local_00 = 0.0f;
    float sigma_local_01 = 0.0f;
    float sigma_local_02 = 0.0f;
    float sigma_local_10 = 0.0f;
    float sigma_local_11 = 0.0f;
    float sigma_local_12 = 0.0f;
    float sigma_local_20 = 0.0f;
    float sigma_local_21 = 0.0f;
    float sigma_local_22 = 0.0f;

    float f_local[3] = { 0.0f };

    if (iter < maxij) {
        int iat, jat;
        ij_at_linij(iter, iat, jat);

        if (iat == jat) {
            const float rcov_sum = rcov[type[iat]] * 2.0f;
            const float dc6i_sum = dc6i[iat];

            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_cn[k-2];
                const int idx2 = tau_idx_cn[k-1];
                const int idx3 = tau_idx_cn[k];

                if (idx1 == rep_cn[0] && idx2 == rep_cn[1] && idx3 == rep_cn[2]) { continue; }
                const float rij[3] = {
                    tau_cn[idx1][idx2][idx3][0],
                    tau_cn[idx1][idx2][idx3][1],
                    tau_cn[idx1][idx2][idx3][2],
                };
                const float r2 = lensq3(rij);
                if (r2 >= cn_thr) { continue; }

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

                sigma_local_00 += vec[0] * rij[0];
                sigma_local_01 += vec[0] * rij[1];
                sigma_local_02 += vec[0] * rij[2];
                sigma_local_10 += vec[1] * rij[0];
                sigma_local_11 += vec[1] * rij[1];
                sigma_local_12 += vec[1] * rij[2];
                sigma_local_20 += vec[2] * rij[0];
                sigma_local_21 += vec[2] * rij[1];
                sigma_local_22 += vec[2] * rij[2];
            }
        }

        else {
            const float rcov_sum = rcov[type[iat]] + rcov[type[jat]];
            const float dc6i_sum = dc6i[iat] + dc6i[jat];

            for (int k = maxtau - 1; k >= 0; k -= 3) {
                const int idx1 = tau_idx_cn[k-2];
                const int idx2 = tau_idx_cn[k-1];
                const int idx3 = tau_idx_cn[k];

                const float rij[3] = {
                    x[jat][0] - x[iat][0] + tau_cn[idx1][idx2][idx3][0],
                    x[jat][1] - x[iat][1] + tau_cn[idx1][idx2][idx3][1],
                    x[jat][2] - x[iat][2] + tau_cn[idx1][idx2][idx3][2]
                };
                const float r2 = lensq3(rij);
                if (r2 >= cn_thr) { continue; }

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

                sigma_local_00 += vec[0] * rij[0];
                sigma_local_01 += vec[0] * rij[1];
                sigma_local_02 += vec[0] * rij[2];
                sigma_local_10 += vec[1] * rij[0];
                sigma_local_11 += vec[1] * rij[1];
                sigma_local_12 += vec[1] * rij[2];
                sigma_local_20 += vec[2] * rij[0];
                sigma_local_21 += vec[2] * rij[1];
                sigma_local_22 += vec[2] * rij[2];
            }
            atomicAdd(&f[iat][0], f_local[0]);
            atomicAdd(&f[iat][1], f_local[1]);
            atomicAdd(&f[iat][2], f_local[2]);
            atomicAdd(&f[jat][0], -f_local[0]);
            atomicAdd(&f[jat][1], -f_local[1]);
            atomicAdd(&f[jat][2], -f_local[2]);
        }
    }

    sigma_00[threadIdx.x] = sigma_local_00;
    sigma_01[threadIdx.x] = sigma_local_01;
    sigma_02[threadIdx.x] = sigma_local_02;
    sigma_10[threadIdx.x] = sigma_local_10;
    sigma_11[threadIdx.x] = sigma_local_11;
    sigma_12[threadIdx.x] = sigma_local_12;
    sigma_20[threadIdx.x] = sigma_local_20;
    sigma_21[threadIdx.x] = sigma_local_21;
    sigma_22[threadIdx.x] = sigma_local_22;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sigma_00[threadIdx.x] += sigma_00[threadIdx.x + s];
            sigma_01[threadIdx.x] += sigma_01[threadIdx.x + s];
            sigma_02[threadIdx.x] += sigma_02[threadIdx.x + s];
            sigma_10[threadIdx.x] += sigma_10[threadIdx.x + s];
            sigma_11[threadIdx.x] += sigma_11[threadIdx.x + s];
            sigma_12[threadIdx.x] += sigma_12[threadIdx.x + s];
            sigma_20[threadIdx.x] += sigma_20[threadIdx.x + s];
            sigma_21[threadIdx.x] += sigma_21[threadIdx.x + s];
            sigma_22[threadIdx.x] += sigma_22[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&sigma[0][0], sigma_00[0]);
        atomicAdd(&sigma[0][1], sigma_01[0]);
        atomicAdd(&sigma[0][2], sigma_02[0]);
        atomicAdd(&sigma[1][0], sigma_10[0]);
        atomicAdd(&sigma[1][1], sigma_11[0]);
        atomicAdd(&sigma[1][2], sigma_12[0]);
        atomicAdd(&sigma[2][0], sigma_20[0]);
        atomicAdd(&sigma[2][1], sigma_21[0]);
        atomicAdd(&sigma[2][2], sigma_22[0]);
    }
}

void PairD3::get_forces_with_dC6() {
    int n = atom->natoms;
    int maxij = n * (n + 1) / 2;
    int maxtau = tau_idx_cn_total_size;

    //START_CUDA_TIMER();

    int threadsPerBlock = 128;
    int blocksPerGrid = (maxij + threadsPerBlock - 1) / threadsPerBlock;
    kernel_get_forces_with_dC6<<<blocksPerGrid, threadsPerBlock>>>(
        maxij, maxtau, cn_thr, K1,
        dc6i, rcov, rep_cn, tau_cn, tau_idx_cn, atomtype, x,
        f, sigma
    );
    cudaDeviceSynchronize();

    //STOP_CUDA_TIMER("get_forces_with");
    //CHECK_CUDA_ERROR();
}


/* ----------------------------------------------------------------------
   Update energy, force, and stress
------------------------------------------------------------------------- */

void PairD3::update(int eflag, int vflag) {
    int n = atom->natoms;

    if (eflag) { eng_vdwl += disp_total * AU_TO_EV; } // Energy update

    double** f_local = atom->f; // Local force of atoms
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f_local[i][j] += f[i][j] * AU_TO_EV / AU_TO_ANG;
        }
    }


    if (vflag) {
        virial[0] += sigma[0][0] * AU_TO_EV;
        virial[1] += sigma[1][1] * AU_TO_EV;
        virial[2] += sigma[2][2] * AU_TO_EV;
        virial[3] += sigma[0][1] * AU_TO_EV;
        virial[4] += sigma[0][2] * AU_TO_EV;
        virial[5] += sigma[1][2] * AU_TO_EV;
    } // Stress update
}

/* ----------------------------------------------------------------------
   Compute : energy, force, and stress (Required)
------------------------------------------------------------------------- */

void PairD3::compute(int eflag, int vflag) {
    if (eflag || vflag)          { ev_setup(eflag, vflag); }
    if (atom->natoms != n_save)  { reallocate_arrays(); }

    set_lattice_vectors();
    precalculate_tau_array();
    load_atom_info();
    cudaMemcpy(atomtype, atom->type, atom->natoms * sizeof(int), cudaMemcpyHostToDevice);

    get_coordination_number();

    int zero_damping = 1;
    int zero_damping_modified = 3;

    if (damping_type == zero_damping) {
        get_forces_without_dC6_zero_damping();
    }
    else if (damping_type == zero_damping_modified){
        get_forces_without_dC6_zero_damping_modified();
    }
    else {
        get_forces_without_dC6_bj_damping();
    }
    get_forces_with_dC6();
    update(eflag, vflag);

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairD3::init_one(int i, int j) {
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
    // No need to count local neighbor in D3
    /* return std::sqrt(rthr * std::pow(au_to_ang, 2)); */
    return 0.0;
}

/* ----------------------------------------------------------------------
   init specific to this pair style (Optional)
------------------------------------------------------------------------- */

void PairD3::init_style() {
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairD3::write_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairD3::read_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairD3::write_restart_settings(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairD3::read_restart_settings(FILE *fp) {}
