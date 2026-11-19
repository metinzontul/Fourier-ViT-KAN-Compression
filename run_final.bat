@echo off
REM ==========================================================================================
REM  Clean final re-run of every frozen-feature experiment with the final code (one code version,
REM  one results folder). Existing results\ is NOT touched; everything goes to results_final\.
REM  ISIC fine-tuning and the efficiency benchmark are copied (their code did not change);
REM  isic_per_class.py and plot_isic_finetune.py use the copied epoch logs and confusion matrices.
REM  Expected run time on an RTX 4080: about 10 hours.
REM ==========================================================================================
set REV_RESULT_DIR=%~dp0results_final
set LRS=3e-4 1e-3 3e-3 1e-2 3e-2
if not exist "%REV_RESULT_DIR%" mkdir "%REV_RESULT_DIR%"

xcopy /E /I /Y "%~dp0results\isic_finetune" "%REV_RESULT_DIR%\isic_finetune" >nul
copy /Y "%~dp0results\efficiency.csv" "%REV_RESULT_DIR%\" >nul
echo [%date% %time%] copied ISIC fine-tuning results and efficiency.csv

python dataset_descriptors.py || goto :error
python frozen_study.py --preset core --lrs %LRS% || goto :error
python frozen_study.py --preset core --reducers bct_perm --lrs %LRS% || goto :error
python frozen_study.py --preset widths --lrs %LRS% || goto :error
python frozen_study.py --datasets MNIST CIFAR10 EuroSAT --backbones vit_b_16 --reducers none --heads kan_orig mlp_orig --lrs 1e-3 --batch_size 64 || goto :error

python analyze.py || goto :error
python penalty_table.py || goto :error
python make_efficiency_table.py || goto :error
python isic_per_class.py || goto :error
python plot_isic_finetune.py || goto :error
python coverage_analysis.py || goto :error
python plot_latent_spectrum.py || goto :error
python plot_width_sweep.py || goto :error
python make_supplement.py || goto :error
echo [%date% %time%] ALL DONE - results in %REV_RESULT_DIR%
goto :eof

:error
echo [%date% %time%] STOPPED: the last command failed. Re-run run_final.bat; finished runs are skipped.
exit /b 1
