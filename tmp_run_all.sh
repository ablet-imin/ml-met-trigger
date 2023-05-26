
source run_jobs_TCL.sh ../mc15_14TeV/ntuples/mar30/mc15_14TeV.600026.PhH7EG_VBFH125_ZZ4nu_MET75.2S_cellE.Mar30/\
 "user.yabulait.*.STREAM_TREE._0000\(.*\).root" data/ZZ4nu


#TTbar
source run_jobs_TCL.sh ../mc15_14TeV/ntuples/mar30/mc15_14TeV.600012.PhPy8EG_A14_ttbar_r13618.2S_cellE.Mar30/\
 "user.yabulait.*.STREAM_TREE._0000\(.*\).root" data/ttbar

source run_jobs_TCL.sh ../mc15_14TeV/ntuples/mar30/mc15_14TeV.800290.Py8EG_jetjet_JZ0WithSW_r13619.2S_cellE.Mar30/\
  "user.yabulait.*.STREAM_TREE._0000\(.*\).root" data/jetjet  Calo422TopoClusters


 
