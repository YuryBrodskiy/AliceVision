@echo off

setlocal

rem ----------------------------------
rem Locate vcpkg using environment variables falling back to sensible defaults
rem ----------------------------------
set "VcPkgDir=%USERPROFILE%\.vcpkg\vcpkg"
set "VcPkgTriplet=x64-windows"
if defined VCPKG_ROOT if /i not "%VCPKG_ROOT%"=="" set "VcPkgDir=%VCPKG_ROOT%"
if defined VCPKG_DEFAULT_TRIPLET if /i not "%VCPKG_DEFAULT_TRIPLET%"=="" set "VcPkgTriplet=%VCPKG_DEFAULT_TRIPLET%"

set g2o_patch="%cd%\g2o_version_update.patch"


pushd %VcPkgDir%

rem ==============================
rem Upgrade and Install packages.
rem ==============================
set VcPkgLibs= boost-algorithm boost-accumulators boost-atomic boost-container boost-date-time boost-exception boost-filesystem boost-graph boost-log ^
boost-program-options boost-property-tree boost-ptr-container boost-regex boost-serialization boost-system boost-test boost-thread ^
openexr ^
openimageio[libraw] ^
alembic ^
geogram ^
eigen3 ^
ceres[suitesparse] ^
cuda 

echo vcpkg found at %VcPkgDir%...
echo installing %VcPkgLibs% for triplet %VcPkgTriplet%...

call vcpkg upgrade %VcPkgLibs% --no-dry-run --triplet %VcPkgTriplet%
call vcpkg install %VcPkgLibs% --recurse --triplet %VcPkgTriplet%

popd

endlocal & set "VcPkgDir=%VcPkgDir%" & set "VcPkgTriplet=%VcPkgTriplet%"
