#!/usr/bin/tcsh -f

# 1. Check if PYTHONPATH exists; if not, initialize it.
if ( $?PYTHONPATH ) then
    setenv PYTHONPATH ${PYTHONPATH}:`pwd`/builddir
else
    setenv PYTHONPATH `pwd`/builddir
endif

# 2. Run your test
python3 -c "import sphere_ops; print('Import Successful!')"
