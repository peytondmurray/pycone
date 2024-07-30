py.install_sources(
  [
    '__init__.py',
    'stats_pymc.py',
    'stats_emcee.py'
  ],
  subdir: 'pycone/stats',
  pure: false,
)

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

py.extension_module(
  'fast_models',
  ['fast_models.pyx'],
  install: true,
  subdir: 'pycone/stats',
  cython_args: ['--cplus'],
  include_directories: [incdir_numpy],
  override_options : ['cython_language=cpp'],
)

py.extension_module(
  'fast_stats_emcee',
  ['fast_stats_emcee.pyx'],
  install: true,
  subdir: 'pycone/stats',
  cython_args: ['--cplus'],
  include_directories: [incdir_numpy],
  override_options : ['cython_language=cpp'],
)
