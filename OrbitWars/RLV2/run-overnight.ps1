$logFile = "C:\Users\Ruben\Github\Kaggle\OrbitWars\RLV2\overnight_bc_train.log"

Write-Output "======= Started $(Get-Date) =======" | Tee-Object -FilePath $logFile

Write-Output "======= Running 2 Player ========" | Tee-Object -FilePath $logFile -Append
python -u bc.py --players 2 --games 800 --epochs 120 --bs 256 --out bc2.pt 2>&1 |
    Tee-Object -FilePath $logFile -Append

Write-Output "======= Running 4 Player ========" | Tee-Object -FilePath $logFile -Append
python -u bc.py --players 4 --games 800 --epochs 120 --bs 256 --out bc4.pt 2>&1 |
    Tee-Object -FilePath $logFile -Append

Write-Output "======= Finished $(Get-Date) =======" | Tee-Object -FilePath $logFile -Append