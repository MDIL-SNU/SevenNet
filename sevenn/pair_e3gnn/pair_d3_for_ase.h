/*
This code is a skeleton of the LAMMPS pair_style d3 accelerated by CUDA.
All dependencies on LAMMPS have been removed.
The input and output variables are named based on the LAMMPS variables.
*/

#ifndef LMP_PAIR_D3
#define LMP_PAIR_D3

#include <cmath>
#include <string>
#include <vector> // for 'element_table'
#include <algorithm> // for 'atomic_number'
#include <map>
#include <unordered_map>
#include <cuda_runtime.h>

#include "pair_d3_pars.h"

// Removed dependencies to STL
// #include <stdlib.h> -> no more C style functions
// #define _USE_MATH_DEFINES -> no predefined constants

// Removed dependencies to LAMMPS
// #include "pair.h"   -> removed, for construncting pair class.
// #include "utils.h"  -> removed, some float converters.
// #include "atom.h"   -> Atom class to replace it.
// #include "domain.h" -> Domain class to replace it.
// #include "error.h"  -> Error class to replace it.
// #include "comm.h"       -> already no dependency
// #include "neighbor.h"   -> already no dependency
// #include "neigh_list.h" -> already no dependency
// #include "memory.h"     -> already no dependency for CUDA version
// #include "math_extra.h"            -> removed, dot and len3 operations.
// #include "potential_file_reader.h" -> removed, PotentialFileReader

/* --------- Fake class to replace 'LAMMPS' class --------- */
class Atom {
public:
    int natoms;
    int ntypes;
    int* type;
    double** x;
    Atom(int natoms, int ntypes, int* type, double** x) :
        natoms(natoms),
        ntypes(ntypes),
        type(type),
        x(x) {}
    ~Atom() {
        //delete[] type;
        //for (int i = 0; i < natoms; i++) {
        //    delete[] x[i];
        //}
        //delete[] x;
    }
};

class Domain {
public:
    int xperiodic, yperiodic, zperiodic;
    double boxlo[3], boxhi[3];
    double xy, xz, yz;
    Domain(int xperiodic, int yperiodic, int zperiodic, double* boxlo, double* boxhi, double xy, double xz, double yz) :
        xperiodic(xperiodic),
        yperiodic(yperiodic),
        zperiodic(zperiodic),
        xy(xy),
        xz(xz),
        yz(yz) {
        for (int i = 0; i < 3; i++) {
            this->boxlo[i] = boxlo[i];
            this->boxhi[i] = boxhi[i];
        }
    }
    ~Domain() {
    }
};

class Error {
public:
    void all(int flerr, const char* message) {
        printf("Error: %s\n", message);
    }
    Error() {}
    ~Error() {}
};
/* ------------------------------------------------------- */

/* --------- Declaration of fake classes and variables --------- */
#define FLERR 1
//Error* error = nullptr;
//
//int allocated;
//int** setflag;
//double** cutsq;

//Atom* atom = nullptr;
//Domain* domain = nullptr;
//
//double result_E;
//double* result_F = nullptr;
//double result_S[6];

class Pair {
public:
    int allocated;
    Atom* atom;
    Domain* domain;
    double result_E;
    double* result_F;
    double result_S[6];
    Error* error;

    Pair()
        : allocated(0), atom(nullptr), domain(nullptr), result_E(0.0), result_F(nullptr), error(nullptr) {
        std::fill(std::begin(result_S), std::end(result_S), 0.0);
    }

    virtual ~Pair() {
        if (result_F) {
            delete[] result_F;
            result_F = nullptr;
        }
        if (atom) {
            delete atom;
            atom = nullptr;
        }
        if (domain) {
            delete domain;
            domain = nullptr;
        }
        if (error) {
            delete error;
            error = nullptr;
        }
    }
};
/* -------------------------------------------------------------- */

class PairD3 : public Pair {
public:
    PairD3();
    ~PairD3();

    void settings(double rthr, double cnthr, std::string damp_name, std::string func_name);
    void coeff(int* atomic_number);
    void compute();

protected:
    virtual void allocate();

    /* ------- Read parameters ------- */
    int find_atomic_number(std::string&);
    int is_int_in_array(int*, int, int);
    void read_r0ab(int*, int); // void read_r0ab(class LAMMPS*, char*, int*, int);
    void get_limit_in_pars_array(int&, int&, int&, int&);
    void read_c6ab(int*, int); // void read_c6ab(class LAMMPS*, char*, int*, int);

    void setfuncpar_zero();
    void setfuncpar_bj();
    void setfuncpar_zerom();
    void setfuncpar_bjm();
    void setfuncpar();
    /* ------- Read parameters ------- */

    /* ------- Lattice information ------- */
    void set_lattice_repetition_criteria(float, int*);
    void set_lattice_vectors();
    /* ------- Lattice information ------- */

    /* ------- Initialize & Precalculate ------- */
    void load_atom_info();
    void precalculate_tau_array();
    /* ------- Initialize & Precalculate ------- */

    /* ------- Reallocate (when number of atoms changed) ------- */
    void reallocate_arrays();
    void reallocate_arrays_np1();
    /* ------- Reallocate (when number of atoms changed) ------- */

    /* ------- Coordination number ------- */
    void get_coordination_number();
    void get_dC6_dCNij();
    /* ------- Coordination number ------- */

    /* ------- Main workers ------- */
    void get_forces_without_dC6_zero();
    void get_forces_without_dC6_bj();
    void get_forces_without_dC6_zerom();
    void get_forces_without_dC6_bjm();
    void get_forces_without_dC6();
    void get_forces_with_dC6();
    void update();
    /* ------- Main workers ------- */

    /*--------- Constants ---------*/
    static constexpr int MAX_ELEM = 94;          // maximum of the element number
    static constexpr int MAXC = 5;               // maximum coordination number references per element

    static constexpr double AU_TO_ANG = 0.52917726; // conversion factors (atomic unit --> angstrom)
    static constexpr double AU_TO_EV = 27.21138505; // conversion factors (atomic unit --> eV)

    static constexpr float K1 = 16.0;              // global ad hoc parameters
    static constexpr float K3 = -4.0;              // global ad hoc parameters
    /*--------- Constants ---------*/

    /*--------- Parameters to read ---------*/
    int damping;
    std::string functional;
    float* r2r4 = nullptr;             // scale r4/r2 values of the atoms by sqrt(Z)
    float* rcov = nullptr;             // covalent radii
    int* mxc = nullptr;                // How large the grid for c6 interpolation
    float** r0ab = nullptr;            // cut-off radii for all element pairs
    float***** c6ab = nullptr;         // C6 for all element pairs
    float rthr;                        // R^2 distance to cutoff for C calculation
    float cnthr;                       // R^2 distance to cutoff for CN_calculation
    float s6, s8, s18, rs6, rs8, rs18, alp, alp6, alp8, a1, a2; // parameters for D3
    /*--------- Parameters to read ---------*/

    /*--------- Lattice related values ---------*/
    double* lat_v_1 = nullptr;           // lattice coordination vector
    double* lat_v_2 = nullptr;           // lattice coordination vector
    double* lat_v_3 = nullptr;           // lattice coordination vector
    int* rep_vdw = nullptr;              // repetition of cell for calculating D3
    int* rep_cn = nullptr;               // repetition of cell for calculating
    double** sigma = nullptr;            // virial pressure on cell
    /*--------- Lattice related values ---------*/

    /*--------- Per-atom values/arrays ---------*/
    double* cn = nullptr;               // Coordination numbers
    float** x = nullptr;                // Positions
    double** f = nullptr;               // Forces
    double* dc6i = nullptr;             // dC6i(iat) saves dE_dsp/dCN(iat)
    /*--------- Per-atom values/arrays ---------*/

    /*--------- Per-pair values/arrays ---------*/
    float* c6_ij_tot = nullptr;
    float* dc6_iji_tot = nullptr;
    float* dc6_ijj_tot = nullptr;
    /*--------- Per-pair values/arrays ---------*/

    /*---------- Global values ---------*/
    int n_save;                         // to check whether the number of atoms has changed
    int np1_save;                       // to check whether the number of types has changed
    float disp_total;                   // Dispersion energy
    /*---------- Global values ---------*/

    /*--------- For loop over tau (translation of cell) ---------*/
    float**** tau_vdw = nullptr;
    float**** tau_cn = nullptr;
    int* tau_idx_vdw = nullptr;
    int* tau_idx_cn = nullptr;
    int tau_idx_vdw_total_size;
    int tau_idx_cn_total_size;
    /*--------- For loop over tau (translation of cell) ---------*/

    /*--------- For cuda memory transfer (pointerized) ---------*/
    int *atomtype;
    double *disp;
    /*--------- For cuda memory transfer (pointerized) ---------*/
};

#endif // LMP_PAIR_D3
