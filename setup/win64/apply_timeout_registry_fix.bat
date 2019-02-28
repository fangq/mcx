@echo off
echo Patching your registry ...
regedit /s launch_timeout_registry_fix.reg
echo Done

echo "You must reboot the computer for the setting to be affective"
echo "press y to reboot"

set /p option=press y to reboot:

if %option%==y shutdown /r

echo "Please remember to reboot the machine to run mcx"

pause
