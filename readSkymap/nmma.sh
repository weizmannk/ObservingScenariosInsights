#!bin/bash





#BNS

nmma_create_injection --prior-file ~/NMMA/nmma/priors/Bu2019lm.prior --injection-file ~/OBSERVING_SCENARIOS/observing-scenarios-simulations/runs/O4/bns_astro/injections.dat --eos-file  ~/NMMA/nmma/example_files/eos/ALF2.dat --binary-type BNS --n-injection 2500 --original-parameters --extension json --aligned-spin



## ZTF
light_curve_analysis_condor --model Bu2019lm --prior  ~/NMMA/nmma/priors/Bu2019lm.prior --svd-path   ~/NMMA/nmma/svdmodels --outdir outdir_BNS --label injection --injection ./injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --ztf-ToO 180 300 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh

### Rubin
light_curve_analysis_condor --model Bu2019lm --prior  ~/NMMA/nmma/priors/Bu2019lm.prior --svd-path   ~/NMMA/nmma/svdmodels --outdir outdir_BNS --label injection --injection ./injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh


#NSBH

nmma_create_injection --prior-file ~/NMMA/nmma/priors/Bu2019nsbh.prior --injection-file ~/OBSERVING_SCENARIOS/observing-scenarios-simulations/runs/O5/nsbh_astro/injections.dat --eos-file  ~/NMMA/nmma/example_files/eos/ALF2.dat --binary-type NSBH --n-injection 2500 --original-parameters --extension json --aligned-spin


## ZTF
light_curve_analysis_condor --model Bu2019nsbh --prior  ~/NMMA/nmma/priors/Bu2019nsbh.prior --svd-path   ~/NMMA/nmma/svdmodels --outdir outdir_NSBH --label injection --injection ./injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --ztf-ToO 180 300 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh


## Rubin
light_curve_analysis_condor --model Bu2019nsbh --prior  ~/NMMA/nmma/priors/Bu2019nsbh.prior --svd-path   ~/NMMA/nmma/svdmodels --outdir outdir_NSBH --label injection --injection ./injection.json --injection-num 2500 --generation-seed 816 323 364 564 851 --condor-dag-file condor.dag --condor-sub-file condor.sub --bash-file condor.sh
