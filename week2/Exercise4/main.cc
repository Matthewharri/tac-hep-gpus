#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h> 
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

	public:
	Particle();
	// FIXME : Create an additional constructor that takes 4 arguments --> the 4-momentum
	Particle(double pT, double Eta, double Phi, double Energy);
	double   pt, eta, phi, E, m, p[4];
	void     p4(double, double, double, double);
	void     print();
	void     setMass(double);
	double   sintheta();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double pT, double Eta, double Phi, double Energy ){ 
	//FIXME
	pt = pT;
	eta = Eta;
	phi = Phi;
	E = Energy;
	m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;

}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){
	//FIXME
	return sin(2*atan(exp(-eta)));
}

void Particle::p4(double pT, double eta, double phi, double energy){
	//FIXME
	p[0] = energy;
	p[1] = pT*cos(phi);
	p[2] = pT*sin(phi);
	p[3] = pT*sinh(eta);

}

void Particle::setMass(double mass)
{
	// FIXME
	// m = sqrt(p[0]*p[0] - p[1]*p[1] - p[2]*p[2] - p[3]*p[3]);
	m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
	// std::cout << std::endl;
	std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}


class Lepton : public Particle{
	public:
	Lepton() : Particle(){charge = 0;};
	Lepton(double pT, double Eta, double Phi, double Energy) : Particle(pT, Eta, Phi, Energy){
		charge = 0;
		this->p4(pT, Eta, Phi, Energy);
		auto mass = p[0]*p[0] - p[1]*p[1] - p[2]*p[2] - p[3]*p[3];
		this->setMass(sqrt(mass));
	};
	float charge;
	void setCharge(float q);
};

void Lepton::setCharge(float q){
	charge = q;
}

class Jet : public Particle{
	public:
	Jet() : Particle(){HadronFlavour = 1;};
	Jet(double pT, double Eta, double Phi, double Energy) : Particle(pT, Eta, Phi, Energy){
		HadronFlavour = 0;
		this->p4(pT, Eta, Phi, Energy);
		auto mass = p[0]*p[0] - p[1]*p[1] - p[2]*p[2] - p[3]*p[3];
		this->setMass(mass);
	}
	float HadronFlavour;
	void setHadronFlavour(float hf);
};

void Jet::setHadronFlavour(float hf){
	HadronFlavour = hf;
}

int main() {
	
	/* ************* */
	/* Input Tree   */
	/* ************* */

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);
	
	t1->SetBranchAddress("njets",&njets);
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();

	for (Long64_t jentry=0; jentry<100;jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<" Event "<< jentry <<std::endl;	

		//FIX ME
		//loop through jets and leptons
		for(auto i = 0; i < njets; i++){
			Jet jet(jetPt[i], jetEta[i], jetPhi[i], jetE[i]);
			jet.setHadronFlavour(jetHadronFlavour[i]);
			std::cout << "Jet: ";
			std::cout << "HadronFlavour: "<< jet.HadronFlavour << ", ";
			jet.print();
		}

		for(auto i = 0; i < 2 ; i++){
			Lepton lepton(lepPt[i], lepEta[i], lepPhi[i], lepE[i]);
			lepton.setCharge(lepQ[i]);
			std::cout << "Lepton: ";
			std::cout << "Charge: "<< lepton.charge << ", ";
			lepton.print();
		}

	} // Loop over all events

  	return 0;
}
