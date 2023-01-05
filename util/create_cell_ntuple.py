import os, sys
import argparse
from array import array
import ROOT

#open root file
#read tree
#create new tree
#loop over tree
##calculate theta from x,y,z
##calculate cell eta
##store cell ex,ey,eta,phi,x,y,z
#close trees

parser = argparse.ArgumentParser(
    prog = "create_cell_ntuple",
    description = "create secondary ntuple from Edison's ntuple",
)

parser.add_argument('--input_file', help="input root file name")
parser.add_argument('--output', help="output root file name")

args = parser.parse_args()

#read input data
source_file = ROOT.TFile.Open(args.input_file,'read')
source_tree = source_file.myTree

get_theta = '''
    using namespace ROOT;
    typedef ROOT::VecOps::RVec<double> RVecD;
    RVecD getTheta( RVecD &fX,  RVecD  &fY,  RVecD &fZ){
        int N = fX.size();
        RVecD all_theta;
        for(int i=0;i<N; i++){
            TVector3 vec3(fX.at(i), fY.at(i), fZ.at(i));
            all_theta.push_back(vec3.Theta());
        };
        return all_theta;
    };
'''
ROOT.gInterpreter.Declare(get_theta)

#cell->energy()*abs( cell->sinTh() )
get_et = '''
    using namespace ROOT;
    typedef ROOT::VecOps::RVec<double> RVecD;
    RVecD get_et(RVecD &E, RVecD &theta){
        auto sinTh = abs(sin(theta));
        return E*sinTh;
    };
'''
ROOT.gInterpreter.Declare(get_et)

get_ex = '''
    using namespace ROOT;
    typedef ROOT::VecOps::RVec<double> RVecD;
    RVecD get_ex(RVecD &Et, RVecD &phi){
        return Et*cos(phi);
    };
'''
get_ey = '''
    using namespace ROOT;
    typedef ROOT::VecOps::RVec<double> RVecD;
    RVecD get_ey(RVecD &Et, RVecD &phi){
        return Et*sin(phi);
    };
'''
ROOT.gInterpreter.Declare(get_ex)
ROOT.gInterpreter.Declare(get_ey)


rdf = ROOT.RDataFrame("myTree", args.input_file)

augmented_cell = rdf.Define("cell_theta", "getTheta(calo.cell.x,calo.cell.y,calo.cell.z)")\
                .Define("cell_et", "get_et(calo.cell.energy, cell_theta)")\
                .Define("cell_ex", "get_ex(cell_et, calo.cell.phi)")\
                .Define("cell_ey", "get_ey(cell_et, calo.cell.phi)")\
                .Define("cell_phi", "calo.cell.phi")\
                .Define("cell_eta", "calo.cell.eta")\
                .Define("cell_sigma","calo.cell.sigma" )\
                .Define("metTruth_ex", "met.Truth.Int.mpx")\
                .Define("metTruth_ey", "met.Truth.Int.mpy")\
                .Define("metTruth_et", "met.Truth.Int.met")



augmented_clt = augmented_cell.Define("clt_theta", "getTheta(calo.cluster.x,calo.cluster.y,calo.cluster.z)")\
                .Define("clt_et", "get_et(calo.cluster.energy, clt_theta)")\
                .Define("clt_ex", "get_ex(clt_et, calo.cluster.phi)")\
                .Define("clt_ey", "get_ey(clt_et, calo.cluster.phi)")\
                .Define("clt_phi", "calo.cluster.phi")\
                .Define("clt_eta", "calo.cluster.eta")\
                .Define("hltMet_ex", "hlt.met.cell.ex")\
                .Define("hltMet_ey", "hlt.met.cell.ey")\
                .Define("hltMet_et", "hlt.met.cell.met")\
                .Define("hltMetPufit_ex", "hlt.met.pufit.ex")\
                .Define("hltMetPufit_ey", "hlt.met.pufit.ey")\
                .Define("hltMetPufit_et", "hlt.met.pufit.met")




augmented_clt.Snapshot("ntuple", args.output,
    ["cell_theta", "cell_et", "cell_ex", "cell_ey", "cell_phi","cell_sigma",
    "cell_eta", "metTruth_ex", "metTruth_ey", "metTruth_et",
    "clt_et", "clt_ex", "clt_ey", "clt_phi", "clt_eta",
    "hltMet_ex", "hltMet_ey", "hltMet_et", "hltMetPufit_ex",
    "hltMetPufit_ey", "hltMetPufit_et"])
    
    
                




