# Train, evaluate and render figures end to end on Windows (run with the project venv active).
# Output is mirrored to train.log.
$env:PYTHONIOENCODING = 'utf-8'

somnus train *>&1 | Tee-Object -FilePath train.log
somnus evaluate *>&1 | Tee-Object -FilePath train.log -Append
Rscript r\figures.R *>&1 | Tee-Object -FilePath train.log -Append
Write-Output 'RUN_ALL_FINISHED' | Tee-Object -FilePath train.log -Append
