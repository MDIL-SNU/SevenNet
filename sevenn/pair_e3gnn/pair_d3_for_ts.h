/*
Batched DFT-D3 dispersion correction for TorchSim.
Processes multiple systems in a single batch on GPU.
All dependencies on LAMMPS have been removed.
*/

#ifndef BATCH_PAIR_D3_H
#define BATCH_PAIR_D3_H

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <cuda_runtime.h>

#include "pair_d3_pars.h"

class BatchPairD3 {
public:
    BatchPairD3();
    ~BatchPairD3();

    /* --- Setup (call once or when config changes) --- */
    void settings(double rthr, double cnthr,
                  std::string damp_name, std::string func_name);
    void coeff(int* atomic_numbers, int ntypes);

    /* --- Per-step batched compute --- */
    /* B           : number of systems in the batch
       natoms_each : [B] number of atoms per system
       atomtype    : [N_total] 1-indexed type per atom (into unified table)
       x_flat      : [N_total*3] positions in Angstrom
       cells       : [B*9] row-major cell vectors per system (Angstrom)
       pbc         : [B*3] periodic boundary flags per system
       energy_out  : [B] output energies in eV
       forces_out  : [N_total*3] output forces in eV/Ang
       stress_out  : [B*9] output stress (Voigt: xx,yy,zz,xy,xz,yz mapped to 3x3) in eV
    */
    void compute(int B,
                 const int* natoms_each,
                 const int* atomtype,
                 const double* x_flat,
                 const double* cells,
                 const int* pbc,
                 double* energy_out,
                 double* forces_out,
                 double* stress_out);

    void compute_async(int B,
                       const int* natoms_each,
                       const int* atomtype,
                       const double* x_flat,
                       const double* cells,
                       const int* pbc);

    void sync(int B,
              const int* natoms_each,
              double* energy_out,
              double* forces_out,
              double* stress_out);

protected:
    /* --- D3 parameter reading (shared across batch) --- */
    int find_atomic_number(std::string&);
    int is_int_in_array(int*, int, int);
    void read_r0ab(int*, int);
    void get_limit_in_pars_array(int&, int&, int&, int&);
    void read_c6ab(int*, int);

    void setfuncpar_zero();
    void setfuncpar_bj();
    void setfuncpar_zerom();
    void setfuncpar_bjm();
    void setfuncpar();

    /* --- Per-system lattice helpers (CPU) --- */
    void compute_lattice_reps(const double lat_v[3][3], const int pbc_flags[3],
                              float r_threshold, int rep_out[3]);
    void compute_tau_flat(const double lat_v[3][3], const int rep[3],
                          float* tau_buf, int n_tau, int& center_idx);

    /* --- Memory management --- */
    void ensure_capacity(int B, int N_total, int64_t P_total,
                         int T_vdw_total, int T_cn_total);
    void free_param_arrays();
    void alloc_param_arrays(int np1);

    /*--------- Constants ---------*/
    static constexpr int MAX_ELEM = 94;
    static constexpr int MAXC = 5;
    static constexpr double AU_TO_ANG = 0.52917726;
    static constexpr double AU_TO_EV = 27.21138505;
    static constexpr float K1 = 16.0f;
    static constexpr float K3 = -4.0f;

    /*--------- D3 parameters (shared, from coeff) ---------*/
    int damping;
    std::string functional;
    float* r2r4 = nullptr;       // [np1]
    float* rcov = nullptr;       // [np1]
    int* mxc = nullptr;          // [np1]
    float** r0ab = nullptr;      // [np1][np1]
    float***** c6ab = nullptr;   // [np1][np1][MAXC][MAXC][3]
    float rthr;
    float cnthr;
    float s6, s8, s18, rs6, rs8, rs18, alp, alp6, alp8, a1, a2;
    int np1_save = 0;            // current allocated size
    bool params_set = false;

    /*--------- Batch flat arrays (managed memory, high-water-mark) ---------*/
    // Offset arrays [B+1]
    int* atom_offset = nullptr;
    int64_t* pair_offset = nullptr;
    int* tau_vdw_offset = nullptr;
    int* tau_cn_offset = nullptr;
    int* center_tau_vdw = nullptr;  // [B]
    int* center_tau_cn = nullptr;   // [B]

    // Per-atom [N_total]
    int* d_atomtype = nullptr;
    float* d_x = nullptr;           // [N_total*3] positions in Bohr
    double* d_cn = nullptr;         // [N_total]
    double* d_dc6i = nullptr;       // [N_total]
    double* d_f = nullptr;          // [N_total*3] forces in AU

    // Per-pair [P_total]
    float* d_c6_ij_tot = nullptr;
    float* d_dc6_iji_tot = nullptr;
    float* d_dc6_ijj_tot = nullptr;

    // Per-system [B]
    double* d_disp = nullptr;       // [B] energy in AU
    double* d_sigma = nullptr;      // [B*9] stress in AU

    // Tau arrays (concatenated)
    float* d_tau_vdw = nullptr;     // [T_vdw_total*3]
    float* d_tau_cn = nullptr;      // [T_cn_total*3]

    /*--------- High-water-mark tracking ---------*/
    int alloc_B = 0;
    int alloc_N = 0;
    int64_t alloc_P = 0;
    int alloc_T_vdw = 0;
    int alloc_T_cn = 0;

    /*--------- CUDA stream ---------*/
    cudaStream_t stream;
};

#endif // BATCH_PAIR_D3_H
