#pragma once

#include <particle_structure.hpp>
#include <particle_structs.hpp>
namespace ps = particle_structs;

namespace {
  // print the contents of a view for debugging
  template <typename ppView>
  void printView(ppView v){
      //printf("view: %s\n", v.label().c_str());
      Kokkos::parallel_for("print_view",
          v.size(),
          KOKKOS_LAMBDA (const int& i) {
            printf("%d %d\n", i, v[i]);
          });
  }

 // print the contents of a broken down MTview for debugging
  template <typename ppView,typename ppView1,typename ppView2>
  void printView(ppView v0, ppView1 v1, ppView2 v2){
      //printf("view: %s\n", v.label().c_str());
      Kokkos::parallel_for("print_view",
          v0.size(),
          KOKKOS_LAMBDA (const int& i) {
            printf("%d %d %d %d\n", i, v0(i), v1(i), v2(i));
          });
  }

  // count the number of elements with particles
  template <typename ppView>
  int countElmsWithPtcls(ppView particles_per_element){
    int count;
    Kokkos::parallel_reduce("count_elements_with_particles",
        particles_per_element.size(),
        KOKKOS_LAMBDA (const int& i, int& lsum ) {
          //SS0 use a kokkos parallel_reduce to count the number of elements
          //that have at least one particle
	  if(particles_per_element( i ) > 0) {
            lsum += 1;
	  }
        }, count);
    return count;
  }

  //reassign particle elements to check if they get assigned properly in 
  //initCrsData as an existing test currently does not exist
  template <typename ppView>
  void reassignPtclElems(ppView particle_elements,ppView particles_per_element,int flag){
    if(flag == 1){
      Kokkos::parallel_for("testing init 1", particle_elements.size(), KOKKOS_LAMBDA(const int& i){
        particle_elements(i) = i%5;
      });
    }
    else if(flag == 2){ //uneven distribution of particles
      Kokkos::parallel_for("testing init 2", particle_elements.size(), KOKKOS_LAMBDA(const int& i){
        if(i%10 > 4) particle_elements(i) = 4;
        else particle_elements(i) = i%5;
      });
      Kokkos::parallel_for("new elem number", particles_per_element.size(),
          KOKKOS_LAMBDA(const int& i){
        if(i == 0) particles_per_element(i) = 3;
        if(i == 1) particles_per_element(i) = 3;
        if(i == 2) particles_per_element(i) = 3;
        if(i == 3) particles_per_element(i) = 3;
        if(i == 4) particles_per_element(i) = 13;
      });
    }
    else if(flag == 3){ //some empty elements
      Kokkos::parallel_for("testing init 3", particle_elements.size(), KOKKOS_LAMBDA(const int& i){
        if(i > 21) particle_elements(i) = 0;
        else if(i > 18) particle_elements(i) = 1;
        else if(i > 12) particle_elements(i) = 3;
        else particle_elements(i) = 4;
      });
      Kokkos::parallel_for("new elem number", particles_per_element.size(),
          KOKKOS_LAMBDA(const int& i){
        if(i == 0) particles_per_element(i) = 3;
        if(i == 1) particles_per_element(i) = 3;
        if(i == 2) particles_per_element(i) = 0;
        if(i == 3) particles_per_element(i) = 6;
        if(i == 4) particles_per_element(i) = 13;
      });
    }
  }
}


namespace pumipic {

  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CSR : public ParticleStructure<DataTypes, MemSpace> {
  public:
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;

    typedef Kokkos::TeamPolicy<execution_space> PolicyType;

    CSR() = delete;
    CSR(const CSR&) = delete;
    CSR& operator=(const CSR&) = delete;

    CSR(PolicyType& p,
        lid_t num_elements, lid_t num_particles, 
        kkLidView particles_per_element,
        kkGidView element_gids, 
        kkLidView particle_elements = kkLidView(),
        MTVs particle_info = NULL);
    ~CSR();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;

    kkLidView getOffsets() { return offsets; }
    MTVs getPtcl_data() { return ptcl_data; }

    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string name="");

    void printMetrics() const;

    //---Attention User---  Do **not** call this function! {
    /**
     * (in) particle_elements - particle_elements[i] contains the id (index) 
     *                          of the parent element * of particle i
     * (in) particle_info - 'member type views' containing the user's data to be
     *                      associated with each particle
     */
    void initCsrData(kkLidView particle_elements, MTVs particle_info) {
      //Create the 'particle_indices' array.  particle_indices[i] stores the 
      //location in the 'ptcl_data' where  particle i is stored.  Use the
      //CSR offsets array and an atomic_fetch_add to compute these entries.
      lid_t given_particles = particle_elements.size();
      assert(given_particles == num_ptcls);

      // create a pointer to the offsets array that we can access in a kokkos parallel_for
      auto offset_cpy = offsets; 
      kkLidView particle_indices("particle_indices", num_ptcls);
      //SS3 insert code to set the entries of particle_indices>
      kkLidView row_indices("row indces", num_elems+1);
      Kokkos::deep_copy(row_indices, offset_cpy);

      Kokkos::parallel_for("particle indices", given_particles, KOKKOS_LAMBDA(const int& i){
        particle_indices(i) = Kokkos::atomic_fetch_add(&row_indices(particle_elements(i)), 1);
      });

      CopyViewsToViews<kkLidView, DataTypes>(ptcl_data, particle_info, particle_indices);
    }
    // } ... or else!

  private:
    //The User defined kokkos policy
    PolicyType policy;

    //Variables from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::num_elems;
    using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity_;
    using ParticleStructure<DataTypes, MemSpace>::num_rows;
    using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
    using ParticleStructure<DataTypes, MemSpace>::num_types;
  
    //Offsets array into CSR
    kkLidView offsets;
  };


  template <class DataTypes, typename MemSpace>
  CSR<DataTypes, MemSpace>::CSR(PolicyType& p,
                                lid_t num_elements, lid_t num_particles,
                                kkLidView particles_per_element,
                                kkGidView element_gids,      //optional
                                kkLidView particle_elements, //optional
                                MTVs particle_info) :        //optional
      ParticleStructure<DataTypes, MemSpace>(), 
      policy(p)
  {
    Kokkos::Profiling::pushRegion("csr_construction");
    num_elems = num_elements;
    num_rows = num_elems;
    num_ptcls = num_particles;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if(!comm_rank)
      fprintf(stderr, "Building CSR\n");

    //new particles per element per flag into the reassign
    int flag = 0;
    //Reassigning elements to check its working right
    reassignPtclElems(particle_elements,particles_per_element,flag);

    //SS1 allocate the offsets array and use an exclusive_scan (aka prefix sum)
    //to fill the entries of the offsets array.
    //see pumi-pic/support/SupportKK.h for the exclusive_scan helper function
    offsets = kkLidView("offsets", num_elems+1); 
    Kokkos::resize(particles_per_element, particles_per_element.size()+1);
    exclusive_scan(particles_per_element, offsets);

    //SS2 set the 'capacity_' of the CSR storage from the last entry of offsets
    //pumi-pic/support/SupportKK.h has a helper function for this
    capacity_ = getLastValue(offsets);
    //allocate storage for user particle data
    CreateViews<device_type, DataTypes>(ptcl_data, capacity_); 

    //printView(offsets);
  

    //Checking input data to verify against once placed into member variable 
    //if(particle_info != NULL){
    //  auto pIDs  = ps::getMemberView<DataTypes,0>(particle_info);
    //  auto vals2 = ps::getMemberView<DataTypes,2>(particle_info);

    //  printView(pIDs,vals2,particle_elements);
    //}


    //If particle info is provided then enter the information
    lid_t given_particles = particle_elements.size();
    if (given_particles > 0 && particle_info != NULL) {
      if(!comm_rank) fprintf(stderr, "initializing CSR data\n");
      initCsrData(particle_elements, particle_info);
    }

    //print to inspect if placed in correct slots
  //  auto pIDs  = ps::getMemberView<DataTypes,0>(ptcl_data);
  //  printView(pIDs);

    if(!comm_rank)
      fprintf(stderr, "Building CSR done\n");
    Kokkos::Profiling::popRegion();
  }

  template <class DataTypes, typename MemSpace>
  CSR<DataTypes, MemSpace>::~CSR() {
    destroyViews<DataTypes, memory_space>(ptcl_data);
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    fprintf(stderr, "[WARNING] CSR migrate(...) not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    fprintf(stderr, "[WARNING] CSR rebuild(...) not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  template <typename FunctionType>
  void CSR<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string name) {
    if (nPtcls() == 0)
      return;
    FunctionType* fn_d;
#ifdef PP_USE_CUDA
    cudaMalloc(&fn_d, sizeof(FunctionType));
    cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
#else
    fn_d = &fn;
#endif
    const lid_t league_size = num_elems;
    const lid_t team_size = 32;  //hack
    const PolicyType policy(league_size, team_size);
    auto offsets_cpy = offsets;
    const lid_t mask = 1; //all particles are active
    Kokkos::parallel_for(name, policy,
        KOKKOS_LAMBDA(const typename PolicyType::member_type& thread) {
        const lid_t elm = thread.league_rank();
        const lid_t start = offsets_cpy(elm);
        const lid_t end = offsets_cpy(elm+1);
        const lid_t numPtcls = end-start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, numPtcls), [=] (lid_t& j) {
          const lid_t particle_id = start+j;
          (*fn_d)(elm, particle_id, mask);
        });
    });
  }

  template <class DataTypes, typename MemSpace>
  void CSR<DataTypes, MemSpace>::printMetrics() const {
    fprintf(stderr, "csr capacity %d\n", capacity_);
  }
}
