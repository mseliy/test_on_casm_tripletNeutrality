#ifndef CASM_ChargeNeutralGrandCanonicalEvent_HH
#define CASM_ChargeNeutralGrandCanonicalEvent_HH
#include <vector>
#include "casm/external/Eigen/Dense"
#include "casm/CASM_global_definitions.hh"
#include "casm/monte_carlo/DoFMod.hh"
#include <iostream>

namespace CASM{

/// \brief Data structure for storing information regarding a proposed charge neutral grand canonical Monte Carlo event
/// Zeyu: this is used as a data framework to store all informations related to charge neutral GCMC
/// Proposing a ChargeNeutralGrandCanonicalEvent will propose 3 GrandCanonicalEvent:
/// A event will be proposed: pick two sites, one Na/Va site and one Si/P site, and apply the same to_value() value in OccMod
/// in this case the charge is always balanced
/// Yan: modify from Jerry's for neutralizing 3-axes system

class ChargeNeutralGrandCanonicalEvent {
	public:
		typedef Index size_type;
		//Construct the event
		ChargeNeutralGrandCanonicalEvent(){
		};

    	/// \brief Constructor
    	///
    	/// \param Nspecies The number of different molecular species in this calculation (use CompositionConverter::components().size())
    	/// \param Ncorr The total number of correlations that could be calculated (use Clexulator::corr_size)
    	///
    	ChargeNeutralGrandCanonicalEvent(size_type Nspecies, size_type Ncorr);

    	/// \brief Set the change in (extensive) formation energy associated with this event
		void set_dEf(double dEf);

    	/// \brief Return change in (extensive) formation energy associated with this event
		std::vector<double> dEf() const;

    	/// \brief Access change in number of species per supercell. Order as in CompositionConverter::components().
    	std::vector<Eigen::VectorXl> &dN();

    	/// \brief const Access change in number of species per supercell. Order as in CompositionConverter::components().
    	const std::vector<Eigen::VectorXl> &dN() const;

    	/// \brief Set the change in number of species in supercell. Order as in CompositionConverter::components().
    	void set_dN(size_type species_type_index, long int dn);

    	/// \brief Return change in number of species in supercell. Order as in CompositionConverter::components().
    	long int dN(size_type species_type_index) const;

    	/// \brief Set change in (extensive) potential energy, dEpot = dEf - sum_i(Nunit * param_chem_pot_i * dcomp_x_i)
    	void set_dEpot(double dpot_nrg);

    	/// \brief Return change in (extensive) potential energy, dEpot = dEf - sum_i(Nunit * param_chem_pot_i * dcomp_x_i)
    	std::vector<double> dEpot() const;

    	/// \brief  Access the occupational modification for this event
        std::vector<OccMod> &occupational_change();

    	/// \brief const Access the occupational modification for this event
    	const std::vector<OccMod> &occupational_change() const;

    	/// \brief Access the changes in (extensive) correlations associated with this event
    	std::vector<Eigen::VectorXd> &dCorr();

    	/// \brief const Access the changes in (extensive) correlations associated with this event
    	const std::vector<Eigen::VectorXd> &dCorr() const;

		void set_original_occ_first_swap(int occ);
		int const original_occ_first_swap() const;

    	void set_current_swap(int current_swap);
    	int const current_swap() const;
 
		void set_dEpot_swapped_twice(double dEpot_swapped_twice);
		double dEpot_swapped_twice();
		const double dEpot_swapped_twice() const;

		void set_dEpot_swapped_threetimes(double dEpot_swapped_threetimes);
		double dEpot_swapped_threetimes();
		const double dEpot_swapped_threetimes() const;

  	private:
    	/// \brief Change in (extensive) correlations due to this event
    	std::vector<Eigen::VectorXd> m_dCorr;
		
    	/// \brief Change in (extensive) formation energy due to this event
    	std::vector<double> m_dEf;

    	/// \brief Change in (extensive) potential energy, dEpot = dEf - sum_i(Nunit * param_chem_pot_i * dcomp_x_i)
    	std::vector<double> m_dEpot;

    	/// \brief Change in number of each species in supercell due to this event.
    	///        The order is determined by primclex.get_param_comp().get_components()
    	std::vector<Eigen::VectorXl> m_dN;

    	/// \brief The ConfigDoF modification performed by this event , Pairs
    	std::vector <OccMod> m_occ_mod;

		int m_current_swap;

		/// dEpot for three swaps
		double m_dEpot_swapped_threetimes;
		double m_dEpot_swapped_twice;
		int m_original_occ_first_swap;		
};

  /// \brief Constructor
  ///
  /// \param Nspecies The number of different molecular species in this calculation (use CompositionConverter::components().size())
  /// \param Ncorr The total number of correlations that could be calculated (use Clexulator::corr_size)
  ///
  inline ChargeNeutralGrandCanonicalEvent::ChargeNeutralGrandCanonicalEvent(size_type Nspecies, size_type Ncorr){
		// if (!is_swapped()){
			m_dCorr[0] = Eigen::VectorXd(Ncorr);
			m_dN[0] = Eigen::VectorXl(Nspecies);
			m_dCorr[1] = Eigen::VectorXd(Ncorr);
			m_dN[1] = Eigen::VectorXl(Nspecies);
			m_dCorr[2] = Eigen::VectorXd(Ncorr);
			m_dN[2] = Eigen::VectorXl(Nspecies);

		// }
		// if (is_swapped()){ // for initialization....
		// }
	 }

	  /// \brief Return change in total (formation) energy associated with this event
	  inline std::vector<double> ChargeNeutralGrandCanonicalEvent::dEf() const {
	    return m_dEf;
	  }
	  /// \brief Set the change in total (formation) energy associated with this event
	  inline void ChargeNeutralGrandCanonicalEvent::set_dEf(double dEf) {
		switch (current_swap()) {
  			case 0:
    			m_dEf[0] = dEf;
    			break;
  			case 1:
    			m_dEf[1] = dEf;
    			break;
  			case 2:
    			m_dEf[2] = dEf;
    			break;
  			default:
    			throw std::runtime_error("Invalid swapped index");
			}
	  }

	  /// \brief Access change in number of all species (extensive). Order as in CompositionConverter::components().
	  inline std::vector<Eigen::VectorXl> &ChargeNeutralGrandCanonicalEvent::dN() {
	    return m_dN;
	  }
	  /// \brief const Access change in number of all species (extensive). Order as in CompositionConverter::components().
	  inline const std::vector<Eigen::VectorXl> &ChargeNeutralGrandCanonicalEvent::dN() const {
	    return m_dN;
	  }

	  /// \brief const Access change in number of species (extensive) described by size_type. Order as in CompositionConverter::components().
	  inline long int ChargeNeutralGrandCanonicalEvent::dN(size_type species_type_index) const {
		switch (current_swap()) {
    		case 0:
        		return m_dN[0](species_type_index);
    		case 1:
        		return m_dN[1](species_type_index);
    		case 2:
        		return m_dN[2](species_type_index);
    		default:
    			throw std::runtime_error("Invalid swapped index");
		}
	  }

	  /// \brief Set the change in number of species (extensive) described by size_type. Order as in CompositionConverter::components().
	  inline void ChargeNeutralGrandCanonicalEvent::set_dN(size_type species_type_index, long int dNi) {
        switch (current_swap()) {
    		case 0:
				m_dN[0](species_type_index) = dNi;
				break;
    		case 1:
        		m_dN[1](species_type_index) = dNi;
				break;
    		case 2:
        		m_dN[2](species_type_index) = dNi;
				break;
    		default:
    			throw std::runtime_error("Invalid swapped index");
		}
	  }

	  /// \brief Set the change in potential energy: dEpot = dEf - sum_i(Nunit * param_chem_pot_i * dcomp_x_i)
	  inline void ChargeNeutralGrandCanonicalEvent::set_dEpot(double dEpot) {
	  	switch (current_swap()) {
    		case 0:
        		m_dEpot[0] = dEpot;
        		break;
    		case 1:
        		m_dEpot[1] = dEpot;
        		break;
    		case 2:
       			m_dEpot[2] = dEpot;
        		break;
    		default:
    			throw std::runtime_error("Invalid swapped index");
	    }
	  }

	  /// \brief Return change in potential energy: dEpot = dEf - sum_i(Nunit * param_chem_pot_i * dcomp_x_i)
	  inline std::vector<double> ChargeNeutralGrandCanonicalEvent::dEpot() const {
	    return m_dEpot;
	  }

	  /// \brief Access the occupational modification for this event
	  inline std::vector<OccMod> &ChargeNeutralGrandCanonicalEvent::occupational_change(){
		  return m_occ_mod;
	  }

	  /// \brief const Access the occupational modification for this event
	  inline const std::vector<OccMod> &ChargeNeutralGrandCanonicalEvent::occupational_change() const{
		  return m_occ_mod;
	  }
	
	  /// \brief Access the changes in (extensive) correlations associated with this event
      inline std::vector<Eigen::VectorXd>&ChargeNeutralGrandCanonicalEvent::dCorr(){
		  return m_dCorr;
	  }

      /// \brief const Access the changes in (extensive) correlations associated with this event
      inline const std::vector<Eigen::VectorXd> &ChargeNeutralGrandCanonicalEvent::dCorr() const{
		  return m_dCorr;
	  }

//	  inline void ChargeNeutralGrandCanonicalEvent::set_is_swapped(bool is_swapped){
//		  m_is_swapped = is_swapped;
//	  }
//	  inline bool ChargeNeutralGrandCanonicalEvent::is_swapped() {
//	    return m_is_swapped;
//	  }	
//	  inline const bool ChargeNeutralGrandCanonicalEvent::is_swapped() const{
//	    return m_is_swapped;
//	  }	

	  inline void ChargeNeutralGrandCanonicalEvent::set_current_swap(int current_swap) {
    	m_current_swap = current_swap;
		}
      inline const int ChargeNeutralGrandCanonicalEvent::current_swap() const {
    	return m_current_swap;
		}

	  inline void ChargeNeutralGrandCanonicalEvent::set_dEpot_swapped_threetimes(double dEpot_swapped_threetimes) {
	    m_dEpot_swapped_threetimes = dEpot_swapped_threetimes;
	  }
	  inline double ChargeNeutralGrandCanonicalEvent::dEpot_swapped_threetimes() {
	    return m_dEpot_swapped_threetimes;
	  }	
	  inline const double ChargeNeutralGrandCanonicalEvent::dEpot_swapped_threetimes() const{
	    return m_dEpot_swapped_threetimes;
	  }	

	  inline void ChargeNeutralGrandCanonicalEvent::set_dEpot_swapped_twice(double dEpot_swapped_twice) {
	    m_dEpot_swapped_twice = dEpot_swapped_twice;
	  }
	  inline double ChargeNeutralGrandCanonicalEvent::dEpot_swapped_twice() {
	    return m_dEpot_swapped_twice;
	  }	
	  inline const double ChargeNeutralGrandCanonicalEvent::dEpot_swapped_twice() const{
	    return m_dEpot_swapped_twice;
	  }	

	  inline void ChargeNeutralGrandCanonicalEvent::set_original_occ_first_swap(int occ){
		  m_original_occ_first_swap = occ;
	  }
	  inline const int ChargeNeutralGrandCanonicalEvent::original_occ_first_swap() const{
		  return m_original_occ_first_swap;
	  }
}
#endif
