
# I used Bu2019lm model and I jioned the priors Bu2019.prior , then the EOS is the 14512.dat which one have been used by Anna to generate the GW posteriors.
# Andrew maybe you could revise the prior to reflect what you expect maybe the ratio_zeta  and alpha ...
#I could write a condor process if needed , and if you expected to run it in CIT or LIGO's clusters.


#1) Create JSON Injection File : GW convertion to KN

nmma-create-injection  --prior-file ./Bu2019lm.prior --injection-file ./bns_O5_injections.dat --eos-file ./example_files/eos/14512.dat --binary-type BNS  --extension json -f ./outdir/injection_Bu2019lm --generation-seed 42 --aligned-spin --eject


#2) Step 3: Light Curve Production

For ZTF telesescope

lightcurve-analysis --model Bu2019lm --svd-path ./svdmodels --interpolation-type sklearn_gp --outdir ./outdir/BNS/0 --label injection_Bu2019lm_0 --prior ./Bu2019lm.prior --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 2048 --Ebv-max 0 --injection ./outdir/injection_Bu2019lm.json --injection-num 0 --injection-detection-limit 21.7,21.4,20.9   --injection-outfile ./outdir/BNS/0/lc.csv --generation-seed 42 --filters ztfg,ztfr,ztfi --plot --remove-nondetections --local-only --ztf-ToO 300  --ztf-uncertainties --ztf-sampling --ztf-ToO 300 


#For Rubin telesescope

lightcurve-analysis --model Bu2019lm --svd-path ./svdmodels --interpolation-type sklearn_gp --outdir ./outdir/BNS/0 --label injection_Bu2019lm_0 --prior ./Bu2019lm.prior --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 2048 --Ebv-max 0 --injection ./outdir/injection_Bu2019lm.json --injection-num 0 --injection-detection-limit 23.9,25.0,24.7,24.0,23.3,22.1   --injection-outfile ./outdir/BNS/0/lc.csv --generation-seed 42 --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__y,ps1__z --plot --remove-nondetections --local-only --rubin-ToO-type BNS --rubin-ToO


