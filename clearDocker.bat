@echo off
echo This will shut down WSL and compact the Docker Desktop virtual disk.
echo Make sure Docker Desktop is fully closed before continuing.
pause

echo Shutting down WSL...
wsl --shutdown

echo Waiting for WSL to fully release the disk...
timeout /t 8 /nobreak >nul

(
echo select vdisk file="%USERPROFILE%\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
echo attach vdisk readonly
echo compact vdisk
echo detach vdisk
echo exit
) | diskpart

echo Done. Press any key to close.
pause