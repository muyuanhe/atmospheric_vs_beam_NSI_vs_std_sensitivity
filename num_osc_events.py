import ROOT
from ROOT import TTree
from ROOT import TFile
from array import array
from ROOT import TCanvas
from ROOT import TH2D
from ROOT import TFile, TDirectory, TTree, TH1D, TH1F, TCanvas, TH2C, TH2F, TH3F, TH3C
from ROOT import gROOT, TCanvas, TF1, TGraph
from ROOT import gStyle
from ROOT import gPad
import numpy as np
ROOT.gSystem.Load('/Users/muyuanh/OscProb/libOscProb.so')

# Number of events = flux * xsec * prob * efficiency=1
# sensitivity = NOE_std - NOE_NSI
# xsec: Enu, area(sigma of order 10E-38)
# Flux: Enu, L, NuE, NuEbar, NuMu, NuMubar


# f_hist = TFile.Open("resolution.root", "recreate")

f1 = TFile.Open("xsec_graphs.root") # cross section root file
    # f1.nu_e_C12, nu_e_bar_C12, nu_mu_C12, nu_mu_bar_C12. branch: tot_cc

f2 = TFile.Open("tree1.root") # atmospheric flux root file
    # tree1.root.t1, branches: Enu, L, NuE, NuEbar, NuMu, NuMubar

# f3 = TFile.Open("g4numiv6_minervame_me000z-200i_98_0006.root") # beam flux

###### beam flux files
### FHC (neutrino mode)
f4 = TFile.Open("flux_FHC_fd_anti_nue.root")
f5 = TFile.Open("flux_FHC_fd_anti_numu.root")
f6 = TFile.Open("flux_FHC_fd_nue.root")
f7 = TFile.Open("flux_FHC_fd_numu.root")
### RHC (anti-neutrino mode)
f8 = TFile.Open("flux_RHC_fd_anti_nue.root")
f9 = TFile.Open("flux_RHC_fd_anti_numu.root")
f10 = TFile.Open("flux_RHC_fd_nue.root")
f11 = TFile.Open("flux_RHC_fd_numu.root")
######




def main(NSI_parameter_list, index):

    c1 = TCanvas('c1', 'sensitivity')
    c1.Divide(2, 2) # show all 4 histograms
    # c1.Divide(1, 2) # only show 2 histograms


    hpxpy1 = TH2D('hpxpy1', 'sensitivity_Ve_to_Vmu', 20, 0, 20000, 20, -1, 1) # bin x, xmin, xmax, bin y, ymin, ymax
    # another way is to use TH2D('hist', 'name; x name;y name', 20, 0, 20000, 20, -1, 1), won't need to SetTitle()
    hpxpy1.SetFillColor( 48 )
    hpxpy1.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy1.GetYaxis().SetTitle("CosZ")
    hpxpy1.GetZaxis().SetTitle("evt / (cm^2 sec sr GeV)")

    hpxpy2 = TH2D('hpxpy2', 'sensitivity_Vmu_to_Ve', 20, 0, 20000, 20, -1, 1) # bin x, xmin, xmax, bin y, ymin, ymax
    hpxpy2.SetFillColor( 48 )
    hpxpy2.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy2.GetYaxis().SetTitle("CosZ")
    hpxpy2.GetZaxis().SetTitle("evt / (cm^2 sec sr GeV)")

    hpxpy3 = TH2D('hpxpy3', 'sensitivity_Vmu_to_Vmu', 20, 0, 20000, 20, -1, 1) # bin x, xmin, xmax, bin y, ymin, ymax
    hpxpy3.SetFillColor( 48 )
    hpxpy3.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy3.GetYaxis().SetTitle("CosZ")
    hpxpy3.GetZaxis().SetTitle("evt / (cm^2 sec sr GeV)")

    hpxpy4 = TH2D('hpxpy4', 'sensitivity_Ve_to_Ve', 20, 0, 20000, 20, -1, 1) # bin x, xmin, xmax, bin y, ymin, ymax
    hpxpy4.SetFillColor( 48 )
    hpxpy4.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy4.GetYaxis().SetTitle("CosZ")
    hpxpy4.GetZaxis().SetTitle("evt / (cm^2 sec sr GeV)")

    ###### Atmospheric ######

    hpxpy5 = TH1F('hpxpy5', 'atmospheric_sensitivity_Ve', 20, 0, 20000) # end phase being Ve
    # hpxpy5.SetFillColor(48)
    hpxpy5.GetXaxis().SetTitle("L/E (km/GeV)") # to compare with beam sensitivity, don't need zenith angle. projection x
    # hpxpy5.GetYaxis().SetTitle("CosZ")
    hpxpy5.GetYaxis().SetTitle("evt")

    hpxpy6 = TH1F('hpxpy6', 'atmospheric_sensitivity_Vmu', 20, 0, 20000)  # end phase being Vmu
    # hpxpy6.SetFillColor(48)
    hpxpy6.GetXaxis().SetTitle("L/E (km/GeV)") # to compare with beam sensitivity, don't need zenith angle. projection x
    # hpxpy6.GetYaxis().SetTitle("CosZ")
    hpxpy6.GetYaxis().SetTitle("evt")

    ###### beam histograms
    ### FHC
    hpxpy7 = TH1F('hpxpy7', 'FHC_beam_sensitivity_Ve', 20, 0, 2000) # end phase Ve
    hpxpy7.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy7.GetYaxis().SetTitle("evt")

    hpxpy8 = TH1F('hpxpy8', 'FHC_beam_sensitivity_Vmu', 20, 0, 2000)  # end phase Vmu
    hpxpy8.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy8.GetYaxis().SetTitle("evt")

    ### RHC
    hpxpy9 = TH1F('hpxpy9', 'RHC_beam_sensitivity_Ve', 20, 0, 2000) # end phase Ve
    hpxpy9.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy9.GetYaxis().SetTitle("evt")

    hpxpy10 = TH1F('hpxpy10', 'RHC_beam_sensitivity_Vmu', 20, 0, 2000)  # end phase Vmu
    hpxpy10.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy10.GetYaxis().SetTitle("evt")

    ### Debug
    hpxpy11 = TH1F('hpxpy11', 'Beam flux NSI', 20, 0, 2000) # same as hpxpy7, just don't substract NSI and std when fill
    hpxpy11.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy11.GetYaxis().SetTitle("evt")

    hpxpy12 = TH1F('hpxpy12', 'Beam flux std', 20, 0, 2000)
    hpxpy12.GetXaxis().SetTitle("L/E (km/GeV)")
    hpxpy12.GetYaxis().SetTitle("evt")



    # set parameters for PMNS_Fast
    p1 = ROOT.OscProb.PMNS_Fast() # PMNS_Base.cxx line 1299 ~ 1328; double PMNS_Base::Prob(int flvi, int flvf, double E, double L)

    # p1.SetMix(th12, th23, th13, delta_cp)
    # p1.SetDeltaMsqrs(dm21, dm32)
    # set parameters for PMNS_NSI
    p2 = ROOT.OscProb.PMNS_NSI()
    # p2.SetIsNuBar = False # by default because assume neutrino
    # p2.SetNSI(eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau, eps_tautau, delta_emu, delta_etau, delta_mutau)
    # p2.SetNSI(0, 0.5, 0, 0, 0, 0, 0, 0, 0)
    p2.SetNSI(NSI_parameter_list[0], NSI_parameter_list[1], NSI_parameter_list[2],
              NSI_parameter_list[3], NSI_parameter_list[4], NSI_parameter_list[5],
              NSI_parameter_list[6], NSI_parameter_list[7], NSI_parameter_list[8])

    # PMNS_base::Prob(int flvi, int flvf, double E, double L), takes exactly 4 arguments
    # flvi is init flavor, flvf is final flavor. PDG code not used: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
    # NOTE: V_e = 0, V_mu = 1, V_tau = 2

    # print(p1.Prob(0, 1, 12.0, 12771.0))
    # print(p2.Prob(0, 1, 12.0, 12771.0))
    for evt_flux in f2.t1: # f2 is atmospheric, looping over atmospheric event; f2 has CosZ, Enu, L, NuE/bar, NuMu/bar
        #######################################
        ###### Cross section calculation ######
        #######################################

        # Define the cross section histograms first
        evt_xsec_e = f1.nu_e_C12.tot_cc # cross-section histogram for electron neutrino
        evt_xsec_mu = f1.nu_mu_C12.tot_cc # xsec histogram for muon neutrino
        evt_xsec_e_bar = f1.nu_e_bar_C12.tot_cc
        evt_xsec_mu_bar = f1.nu_mu_bar_C12.tot_cc

        # For each neutrino energy(Enu), evaluate the specific cross section using TGraph.Eval(x)
        xsec_e = evt_xsec_e.Eval(evt_flux.Enu)
        xsec_mu = evt_xsec_mu.Eval(evt_flux.Enu)
        xsec_e_bar = evt_xsec_e_bar.Eval(evt_flux.Enu)
        xsec_mu_bar = evt_xsec_mu_bar.Eval(evt_flux.Enu)

        ##########################################################
        ###### Oscillation probability from OscProb package ######
        ##########################################################

        ### Atmospheric (can't differentiate neutrino and anti neutrino)
        # STD
        oscprob_e_to_mu_std = p1.Prob(0, 1, evt_flux.Enu, evt_flux.L) # V_e->V_mu
        oscprob_mu_to_e_std = p1.Prob(1, 0, evt_flux.Enu, evt_flux.L) # V_mu->V_e
        oscprob_e_to_e_std = p1.Prob(0, 0, evt_flux.Enu, evt_flux.L) # V_e->V_e
        oscprob_mu_to_mu_std = p1.Prob(1, 1, evt_flux.Enu, evt_flux.L)  # V_mu->V_mu
        # NSI
        oscprob_e_to_mu_NSI = p2.Prob(0, 1, evt_flux.Enu, evt_flux.L)  # V_e->V_mu
        oscprob_mu_to_e_NSI = p2.Prob(1, 0, evt_flux.Enu, evt_flux.L)  # V_mu->V_e
        oscprob_e_to_e_NSI = p2.Prob(0, 0, evt_flux.Enu, evt_flux.L)  # V_e->V_e
        oscprob_mu_to_mu_NSI = p2.Prob(1, 1, evt_flux.Enu, evt_flux.L)  # V_mu->V_mu



        #########################################################################
        ###### Number of events  = flux * cross_section * oscprob * factor ######
        #########################################################################

        ### atmospheric
        # const double kAna2020FHCLivetime = 555.3415;
        #   const double kAna2020RHCLivetime = 321.1179; Here's the total lifetime of atmospheric, since electronics can't handle 24/7.very short time.
        # adds up to 876s # wrong
        ### that time is wrong. Should be 3e8s

        ######
        selection_efficiency = 0.01
        ######

        # Ve -> Vmu
        NOE_std_e_to_mu = evt_flux.NuE * xsec_mu * oscprob_e_to_mu_std * 1e-38 * 1e-4 * 3e8 * selection_efficiency # NuE * xsec_mu
        NOE_NSI_e_to_mu = evt_flux.NuE * xsec_mu * oscprob_e_to_mu_NSI* 1e-38 * 1e-4 * 3e8  * selection_efficiency# 1e-38 is scaling; 1e-4 is m^2 to cm^2, 3e8 is total beam lifetime
        # Vmu -> Ve
        NOE_std_mu_to_e = evt_flux.NuMu * xsec_e * oscprob_mu_to_e_std * 1e-38 * 1e-4 * 3e8 * selection_efficiency
        NOE_NSI_mu_to_e = evt_flux.NuMu * xsec_e * oscprob_mu_to_e_NSI * 1e-38 * 1e-4 * 3e8 * selection_efficiency
        # Vmu -> Vmu
        NOE_std_mu_to_mu = evt_flux.NuMu * xsec_mu * oscprob_mu_to_mu_std * 1e-38 * 1e-4 * 3e8 * selection_efficiency
        NOE_NSI_mu_to_mu = evt_flux.NuMu * xsec_mu * oscprob_mu_to_mu_NSI * 1e-38 * 1e-4 * 3e8 * selection_efficiency
        # Ve -> Ve
        NOE_std_e_to_e = evt_flux.NuE * xsec_e * oscprob_e_to_e_std * 1e-38 * 1e-4 * 3e8 * selection_efficiency
        NOE_NSI_e_to_e = evt_flux.NuE * xsec_e * oscprob_e_to_e_NSI * 1e-38 * 1e-4 * 3e8 * selection_efficiency





        #######################################
        ###### Sensitivity Histogram Fill ######
        #######################################

        ### Atmospheric
        hpxpy1.Fill(float(evt_flux.L/evt_flux.Enu), float(evt_flux.CosZ), abs(NOE_std_e_to_mu - NOE_NSI_e_to_mu)) # x: L/E, y: cosy
        hpxpy2.Fill(float(evt_flux.L / evt_flux.Enu), float(evt_flux.CosZ), abs(NOE_std_mu_to_e - NOE_NSI_mu_to_e))
        hpxpy3.Fill(float(evt_flux.L / evt_flux.Enu), float(evt_flux.CosZ), abs(NOE_std_mu_to_mu - NOE_NSI_mu_to_mu))
        hpxpy4.Fill(float(evt_flux.L / evt_flux.Enu), float(evt_flux.CosZ), abs(NOE_std_e_to_e - NOE_NSI_e_to_e))
        # ending in Ve, no zenith angle.
        hpxpy5.Fill(float(evt_flux.L / evt_flux.Enu), abs(NOE_std_mu_to_e - NOE_NSI_mu_to_e) *3e35) # mu->e
        hpxpy5.Fill(float(evt_flux.L / evt_flux.Enu), abs(NOE_std_e_to_e - NOE_NSI_e_to_e) *3e35) # e->e
        # ending in Vmu
        hpxpy6.Fill(float(evt_flux.L/evt_flux.Enu), abs(NOE_std_e_to_mu - NOE_NSI_e_to_mu) *3e35) # e->mu
        hpxpy6.Fill(float(evt_flux.L / evt_flux.Enu), abs(NOE_std_mu_to_mu - NOE_NSI_mu_to_mu) *3e35) # mu->mu









    ##################
    ###### Beam ######
    ##################
    ### For loop for beam; Energy range 0 to 10 GeV; While atmospheric range 0 to 10,000 GeV
    for energy in np.arange(0.5, 10, 0.1): # 0 to 10 GeV with 0.1 GeV increment

        #######################################
        ###### Cross section calculation ######
        #######################################

        # Define the cross section histograms first
        evt_xsec_e = f1.nu_e_C12.tot_cc  # cross-section histogram for electron neutrino
        evt_xsec_mu = f1.nu_mu_C12.tot_cc  # xsec histogram for muon neutrino
        evt_xsec_e_bar = f1.nu_e_bar_C12.tot_cc
        evt_xsec_mu_bar = f1.nu_mu_bar_C12.tot_cc
        # For each neutrino energy(Enu), evaluate the specific cross section using TGraph.Eval(x)
        xsec_e = evt_xsec_e.Eval(energy)
        xsec_mu = evt_xsec_mu.Eval(energy)
        xsec_e_bar = evt_xsec_e_bar.Eval(energy)
        xsec_mu_bar = evt_xsec_mu_bar.Eval(energy)

        ##########################################################
        ###### Oscillation probability from OscProb package ######
        ##########################################################

        ### Beam (beam L is fixed to 810 km)(beam has 2 modes, FHC & RHC); default is SetIsNuBar(False)
        # STD neutrino
        p1.SetIsNuBar(False)  # p1 is std, False means neutrino
        beam_oscprob_e_to_mu_std = p1.Prob(0, 1, energy, 810)  # V_e->V_mu
        beam_oscprob_mu_to_e_std = p1.Prob(1, 0, energy, 810)  # V_mu->V_e
        beam_oscprob_e_to_e_std = p1.Prob(0, 0, energy, 810)  # V_e->V_e
        beam_oscprob_mu_to_mu_std = p1.Prob(1, 1, energy, 810)  # V_mu->V_mu
        # STD anti-neutrino
        p1.SetIsNuBar(True)  # anti-neutrino standard interaction
        beam_oscprob_e_bar_to_mu_bar_std = p1.Prob(0, 1, energy, 810)  # V_e_bar->V_mu_bar
        beam_oscprob_mu_bar_to_e_bar_std = p1.Prob(1, 0, energy, 810)  # V_mu_bar->V_e_bar
        beam_oscprob_e_bar_to_e_bar_std = p1.Prob(0, 0, energy, 810)  # V_e_bar->V_e_bar
        beam_oscprob_mu_bar_to_mu_bar_std = p1.Prob(1, 1, energy, 810)  # V_mu_bar->V_mu_bar

        # NSI neutrino
        p2.SetIsNuBar(False)  # p2 is NSI
        beam_oscprob_e_to_mu_NSI = p2.Prob(0, 1, energy, 810)  # V_e->V_mu
        beam_oscprob_mu_to_e_NSI = p2.Prob(1, 0, energy, 810)  # V_mu->V_e
        beam_oscprob_e_to_e_NSI = p2.Prob(0, 0, energy, 810)  # V_e->V_e
        beam_oscprob_mu_to_mu_NSI = p2.Prob(1, 1, energy, 810)  # V_mu->V_mu
        p2.SetIsNuBar(True)  # anti-neutrino
        # NSI anti-neutrino
        beam_oscprob_e_bar_to_mu_bar_NSI = p2.Prob(0, 1, energy, 810)  # V_e_bar->V_mu_bar
        beam_oscprob_mu_bar_to_e_bar_NSI = p2.Prob(1, 0, energy, 810)  # V_mu_bar->V_e_bar
        beam_oscprob_e_bar_to_e_bar_NSI = p2.Prob(0, 0, energy, 810)  # V_e_bar->V_e_bar
        beam_oscprob_mu_bar_to_mu_bar_NSI = p2.Prob(1, 1, energy, 810)  # V_mu_bar->V_mu_bar

        ################################################################################################
        ###### Number of events  = flux * cross_section * oscprob * factor * selection_efficiency ######
        ################################################################################################

        ### Beam NOE std/NSI
        ### Scaling: 3e8 seconds for 10 years
        ### for NOE beam, time it by FHC POT #; For ATM, time it by 3e8
        ### POT proton on target
        ### FHC POT: 14.23e20
        ### RHC POT: 12.50e20
        ###### FHC ######
        # 1e-38 is scaling; 1e-4 is m^2 to cm^2
        ### Ve -> Vmu; f6 is nue beam flux
        # std
        FHC_beam_NOE_std_e_to_mu = f6.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_e_to_mu_std * 1e-38 * 1e-4 * 14.23e20 # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_e_to_mu = f6.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_e_to_mu_NSI * 1e-38 * 1e-4 * 14.23e20

        ### Ve_bar -> Vmu_bar; f4 is nue_bar beam flux
        # std
        FHC_beam_NOE_std_e_bar_to_mu_bar = f4.hpx1.GetBinContent(
            f4.hpx1.FindBin(energy)) * xsec_mu_bar * beam_oscprob_e_bar_to_mu_bar_std * 1e-38 * 1e-4 * 14.23e20
        # NSI
        FHC_beam_NOE_NSI_e_bar_to_mu_bar = f4.hpx1.GetBinContent(
            f4.hpx1.FindBin(energy)) * xsec_mu_bar * beam_oscprob_e_bar_to_mu_bar_NSI * 1e-38 * 1e-4 * 14.23e20

        ### Vmu -> Vmu
        # std
        FHC_beam_NOE_std_mu_to_mu = f7.hpx1.GetBinContent(
            f7.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_mu_to_mu_std * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_mu_to_mu = f7.hpx1.GetBinContent(
            f7.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_mu_to_mu_NSI * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu

        ### Vmu_bar -> Vmu_bar
        # std
        FHC_beam_NOE_std_mu_bar_to_mu_bar = f5.hpx1.GetBinContent(f5.hpx1.FindBin(
            energy)) * xsec_mu_bar * beam_oscprob_mu_bar_to_mu_bar_std * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_mu_bar_to_mu_bar = f5.hpx1.GetBinContent(f5.hpx1.FindBin(
            energy)) * xsec_mu_bar * beam_oscprob_mu_bar_to_mu_bar_NSI * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu

        ### Vmu -> Ve
        # std
        FHC_beam_NOE_std_mu_to_e = f6.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_mu_to_e_std * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_mu_to_e = f6.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_mu_to_e_NSI * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu

        ### Vmu_bar -> Ve_bar
        # std
        FHC_beam_NOE_std_mu_bar_to_e_bar = f5.hpx1.GetBinContent(f5.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_mu_bar_to_e_bar_std * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_mu_bar_to_e_bar = f5.hpx1.GetBinContent(f5.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_mu_bar_to_e_bar_NSI * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu

        ### Ve -> Ve
        # std
        FHC_beam_NOE_std_e_to_e = f6.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_e_to_e_std * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_e_to_e = f6.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_e_to_e_NSI * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu

        ### Ve_bar -> Ve_bar
        # std
        FHC_beam_NOE_std_e_bar_to_e_bar = f4.hpx1.GetBinContent(f4.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_e_bar_to_e_bar_std * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu
        # NSI
        FHC_beam_NOE_NSI_e_bar_to_e_bar = f4.hpx1.GetBinContent(f4.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_e_bar_to_e_bar_NSI * 1e-38 * 1e-4 * 14.23e20  # NuE * xsec_mu

        # print("Flux: ", f4.hpx1.FindBin(energy)) # Order of 1000
        # print("xsec_e_bar: ", xsec_e_bar) # Order of 10
        # print("FHC_beam_NOE_NSI_e_bar_to_e_bar: ", FHC_beam_NOE_NSI_e_bar_to_e_bar) # Order of 1e049

        ### RHC

        ### Ve -> Vmu
        # std
        RHC_beam_NOE_std_e_to_mu = f10.hpx1.GetBinContent(
            f10.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_e_to_mu_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_e_to_mu = f10.hpx1.GetBinContent(
            f10.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_e_to_mu_NSI * 1e-38 * 1e-4 * 12.50e20

        ### Ve_bar -> Vmu_bar
        # std
        RHC_beam_NOE_std_e_bar_to_mu_bar = f8.hpx1.GetBinContent(
            f8.hpx1.FindBin(energy)) * xsec_mu_bar * beam_oscprob_e_bar_to_mu_bar_std * 1e-38 * 1e-4 * 12.50e20
        # NSI
        RHC_beam_NOE_NSI_e_bar_to_mu_bar = f8.hpx1.GetBinContent(
            f8.hpx1.FindBin(energy)) * xsec_mu_bar * beam_oscprob_e_bar_to_mu_bar_NSI * 1e-38 * 1e-4 * 12.50e20

        ### Vmu -> Vmu
        # std
        RHC_beam_NOE_std_mu_to_mu = f11.hpx1.GetBinContent(
            f11.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_mu_to_mu_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_mu_to_mu = f11.hpx1.GetBinContent(
            f11.hpx1.FindBin(energy)) * xsec_mu * beam_oscprob_mu_to_mu_NSI * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu

        ### Vmu_bar -> Vmu_bar
        # std
        RHC_beam_NOE_std_mu_bar_to_mu_bar = f9.hpx1.GetBinContent(f9.hpx1.FindBin(
            energy)) * xsec_mu_bar * beam_oscprob_mu_bar_to_mu_bar_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_mu_bar_to_mu_bar = f9.hpx1.GetBinContent(f9.hpx1.FindBin(
            energy)) * xsec_mu_bar * beam_oscprob_mu_bar_to_mu_bar_NSI * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu

        ### Vmu -> Ve
        # std
        RHC_beam_NOE_std_mu_to_e = f11.hpx1.GetBinContent(
            f11.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_mu_to_e_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_mu_to_e = f11.hpx1.GetBinContent(
            f11.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_mu_to_e_NSI * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu

        ### Vmu_bar -> Ve_bar
        # std
        RHC_beam_NOE_std_mu_bar_to_e_bar = f9.hpx1.GetBinContent(f9.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_mu_bar_to_e_bar_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_mu_bar_to_e_bar = f9.hpx1.GetBinContent(f9.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_mu_bar_to_e_bar_NSI * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu

        ### Ve -> Ve
        # std
        RHC_beam_NOE_std_e_to_e = f10.hpx1.GetBinContent(
            f10.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_e_to_e_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_e_to_e = f10.hpx1.GetBinContent(
            f6.hpx1.FindBin(energy)) * xsec_e * beam_oscprob_e_to_e_NSI * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu

        ### Ve_bar -> Ve_bar
        # std
        RHC_beam_NOE_std_e_bar_to_e_bar = f8.hpx1.GetBinContent(f8.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_e_bar_to_e_bar_std * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu
        # NSI
        RHC_beam_NOE_NSI_e_bar_to_e_bar = f8.hpx1.GetBinContent(f8.hpx1.FindBin(
            energy)) * xsec_e_bar * beam_oscprob_e_bar_to_e_bar_NSI * 1e-38 * 1e-4 * 12.50e20  # NuE * xsec_mu

        #######################################
        ###### Sensitivity Histogram Fill ######
        #######################################
        ### total nucleons in detector: 3e35 # 7e34, factored 700 times from ND to FD.
        ### divide by 5e6, since every flux file has 5e5 POT, 10 merged to 1

        ### FHC
        # hpxpy7 end phase Ve, standard
        # print(float(810/energy))
        # print(FHC_beam_NOE_std_e_to_e)
        # print(FHC_beam_NOE_NSI_e_to_e)
        hpxpy7.Fill(float(810 / energy), abs(FHC_beam_NOE_std_e_to_e - FHC_beam_NOE_NSI_e_to_e) *3e35/5e6)
        # print(abs(FHC_beam_NOE_std_e_to_e - FHC_beam_NOE_NSI_e_to_e))
        hpxpy7.Fill(float(810 / energy), abs(FHC_beam_NOE_std_mu_to_e - FHC_beam_NOE_NSI_mu_to_e) *3e35/5e6)
        hpxpy7.Fill(float(810 / energy), abs(FHC_beam_NOE_std_e_bar_to_e_bar - FHC_beam_NOE_NSI_e_bar_to_e_bar) *3e35/5e6)
        hpxpy7.Fill(float(810 / energy), abs(FHC_beam_NOE_std_mu_bar_to_e_bar - FHC_beam_NOE_NSI_mu_bar_to_e_bar) *3e35/5e6)
        # hpxpy8 end phase Vmu
        hpxpy8.Fill(float(810 / energy), abs(FHC_beam_NOE_std_e_to_mu - FHC_beam_NOE_NSI_e_to_mu) *3e35/5e6)
        hpxpy8.Fill(float(810 / energy), abs(FHC_beam_NOE_std_mu_to_mu - FHC_beam_NOE_NSI_mu_to_mu) *3e35/5e6)
        hpxpy8.Fill(float(810 / energy), abs(FHC_beam_NOE_std_e_bar_to_mu_bar - FHC_beam_NOE_NSI_e_bar_to_mu_bar) *3e35/5e6)
        hpxpy8.Fill(float(810 / energy),
                    abs(FHC_beam_NOE_std_mu_bar_to_mu_bar - FHC_beam_NOE_NSI_mu_bar_to_mu_bar) *3e35/5e6)

        ### RHC
        # hpxpy9 end phase Ve
        hpxpy9.Fill(float(810 / energy), abs(RHC_beam_NOE_std_e_to_e - RHC_beam_NOE_NSI_e_to_e) *3e35/5e6)
        hpxpy9.Fill(float(810 / energy), abs(RHC_beam_NOE_std_mu_to_e - RHC_beam_NOE_NSI_mu_to_e) *3e35/5e6)
        hpxpy9.Fill(float(810 / energy), abs(RHC_beam_NOE_std_e_bar_to_e_bar - RHC_beam_NOE_NSI_e_bar_to_e_bar) *3e35/5e6)
        hpxpy9.Fill(float(810 / energy), abs(RHC_beam_NOE_std_mu_bar_to_e_bar - RHC_beam_NOE_NSI_mu_bar_to_e_bar) *3e35/5e6)
        # hpxpy10 end phase Vmu
        hpxpy10.Fill(float(810 / energy), abs(RHC_beam_NOE_std_e_to_mu - RHC_beam_NOE_NSI_e_to_mu) *3e35/5e6)
        hpxpy10.Fill(float(810 / energy), abs(RHC_beam_NOE_std_mu_to_mu - RHC_beam_NOE_NSI_mu_to_mu) *3e35/5e6)
        hpxpy10.Fill(float(810 / energy),
                     abs(RHC_beam_NOE_std_e_bar_to_mu_bar - RHC_beam_NOE_NSI_e_bar_to_mu_bar) *3e35/5e6)
        hpxpy10.Fill(float(810 / energy),
                     abs(RHC_beam_NOE_std_mu_bar_to_mu_bar - RHC_beam_NOE_NSI_mu_bar_to_mu_bar) *3e35/5e6)
















    # c1.cd(1)
    # gPad.SetRightMargin(0.15)
    # gStyle.SetOptStat(0) # No stats
    # hpxpy1.Draw('colz')
    #
    # c1.cd(2)
    # gPad.SetRightMargin(0.15)
    # gStyle.SetOptStat(0)
    # hpxpy2.Draw('colz')
    #
    # c1.cd(3)
    # gPad.SetRightMargin(0.15)
    # gStyle.SetOptStat(0)
    # hpxpy3.Draw('colz')
    #
    # c1.cd(4)
    # gPad.SetRightMargin(0.15)
    # gStyle.SetOptStat(0)
    # hpxpy4.Draw('colz')

    c1.cd(1)
    gPad.SetRightMargin(0.15)
    gStyle.SetOptStat(0)
    hpxpy5.Draw('HIST TEXT0')

    c1.cd(2)
    gPad.SetRightMargin(0.15)
    gStyle.SetOptStat(0)
    hpxpy6.Draw('HIST TEXT0')

    # Note: atmospheric doesn't have FHC or RHC
    # FHC
    # c1.cd(3)
    # gPad.SetRightMargin(0.15)
    # gStyle.SetOptStat(0)
    # # hpxpy7.Draw('HIST TEXT0')
    # hpxpy7.Draw('HIST')
    #
    # c1.cd(4)
    # gPad.SetRightMargin(0.15)
    # gStyle.SetOptStat(0)
    # # hpxpy8.Draw('HIST TEXT0')
    # hpxpy8.Draw('HIST')

    # RHC
    c1.cd(3)
    gPad.SetRightMargin(0.15)
    gStyle.SetOptStat(0)
    # hpxpy9.Draw('HIST TEXT0')
    hpxpy9.Draw('HIST')


    c1.cd(4)
    gPad.SetRightMargin(0.15)
    gStyle.SetOptStat(0)
    # hpxpy10.Draw('HIST TEXT0')
    hpxpy10.Draw('HIST')



    c1.Update()
    # c1.SaveAs("NSI_sensitivity"+str(index)+"RHC.pdf")
    # c1.SaveAs("RHC_NSI_eps_emu05_delta_emu05.pdf")
    # c1.SaveAs("RHC_NSI_eps_etau05_delta_etau05.pdf")
    c1.SaveAs("RHC_NSI_eps_mutau05_deltamutau05.pdf")

    # f_hist.Close()

### This one for eps values
# for i in range(9):
#     L = [0, 0, 0, 0, 0, 0, 0, 0, 0] # eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau, eps_tautau, delta_emu, delta_etau, delta_mutau
#     L[i] = 0.5 # NSI parameter set to 0.5
#     main(L, i)

### This section for delta values
#
# eps_ee, eps_emu, eps_etau, eps_mumu, eps_mutau, eps_tautau, delta_emu, delta_etau, delta_mutau
# L = [0, 0.5, 0, 0, 0, 0, 0.5, 0, 0] # e_mu
# L = [0, 0, 0.5, 0, 0, 0, 0, 0.5, 0] # e_tau
L = [0, 0, 0, 0, 0.5, 0, 0, 0, 0.5] # mu_tau
# eps_ee, eps_emu, eps_etau, eps_mumu, \
# eps_mutau, eps_tautau, delta_emu, delta_etau, delta_mutau
main(L, 0)