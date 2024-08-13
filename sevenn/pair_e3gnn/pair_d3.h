/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(d3, PairD3)

#else

#ifndef LMP_PAIR_D3
#define LMP_PAIR_D3
#define _USE_MATH_DEFINES

#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unordered_map>
#include <omp.h>
#include "pair.h"
#include "memory.h"
#include "atom.h"
#include "utils.h"
#include "error.h"
#include "comm.h"
#include "potential_file_reader.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "domain.h"
#include "math_extra.h"
#include "pair_d3_pars.h"

namespace LAMMPS_NS {

    class PairD3 : public Pair {
    public:
        // Constructor
        PairD3(class LAMMPS*);
        // Destructor
        ~PairD3() override;

        void compute(int, int) override;
        void settings(int, char**) override;
        void coeff(int, char**) override;
        double init_one(int i, int j) override;
        void init_style() override;

        void write_restart(FILE*) override;
        void read_restart(FILE*) override;
        void write_restart_settings(FILE*) override;
        void read_restart_settings(FILE*) override;

    protected:
        virtual void allocate();

        /* ------- Read parameters ------- */
        int find_atomic_number(std::string&);
        int is_int_in_array(int*, int, int);
        void read_r0ab(int*, int);
        void get_limit_in_pars_array(int&, int&, int&, int&);
        void read_c6ab(int*, int);
        void setfuncpar(char*);
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
        /* ------- Reallocate (when number of atoms changed) ------- */


        /* ------- Coordination number ------- */
        void get_coordination_number();
        void get_dC6_dCNij();
        /* ------- Coordination number ------- */


        /* ------- Main workers ------- */
        void get_forces_without_dC6_zero_damping();
        void get_forces_without_dC6_zero_damping_modified();
        void get_forces_without_dC6_bj_damping();
        void get_forces_with_dC6();
        void update(int, int);
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
        int damping_type;
        float* r2r4 = nullptr;             // scale r4/r2 values of the atoms by sqrt(Z)
        float* rcov = nullptr;             // covalent radii
        int* mxc = nullptr;                // How large the grid for c6 interpolation
        float** r0ab = nullptr;            // cut-off radii for all element pairs
        float***** c6ab = nullptr;         // C6 for all element pairs
        float rthr;              // R^2 distance to cutoff for C calculation
        float cn_thr;            // R^2 distance to cutoff for CN_calculation
        float s6, s8, s18, rs6, rs8, rs18, alp, alp6, alp8, a1, a2; // parameters for D3
        /*--------- Parameters to read ---------*/


        /*--------- Lattice related values ---------*/
        double* lat_v_1 = nullptr;           // lattice coordination vector
        double* lat_v_2 = nullptr;           // lattice coordination vector
        double* lat_v_3 = nullptr;           // lattice coordination vector
        int* rep_vdw = nullptr;             // repetition of cell for calculating D3
        int* rep_cn = nullptr;              // repetition of cell for calculating
                                            // coordination number
        double** sigma = nullptr;           // virial pressure on cell
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

    };
}    // namespace LAMMPS_NS

#endif
#endif
