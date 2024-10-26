project(
  'fast_dt',
  'cpp',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

py_mod = import('python')
py = py_mod.find_installation()

pybind11_dep = dependency('pybind11', version: '>=2.10.4')
gsl_dep = dependency('gsl', version: '>=2.8')
boost_dep = dependency('boost', version: '>=1.83')

includes = include_directories(
  [
    'pycone/src',
  ]
)

py.install_sources(
  [
    'pycone/__init__.py',
    'pycone/__main__.py',
    'pycone/analysis.py',
    'pycone/explorer.py',
    'pycone/run.py',
    'pycone/preprocess.py',
    'pycone/output.py',
    'pycone/util.py',
  ],
  subdir: 'pycone',
  pure: false,
)

py.install_sources(
  [
    'pycone/stats/__init__.py',
    'pycone/stats/stats_pymc.py',
    'pycone/stats/stats_emcee.py',
    'pycone/stats/math.py',
    'pycone/stats/model.py',
    'pycone/stats/transform.py',
  ],
  subdir: 'pycone/stats',
  pure: false,
)

install_data(
  'pycone/data/cone_crop.xlsx',
  'pycone/data/daily_weather_1981-2014.xlsx',
  'pycone/data/jalene_compilation_2014_lter.xlsx',
  install_dir: py.get_install_dir(
    subdir: 'pycone/data',
    pure: false,
  )
)

py.extension_module(
  '_pycone_main',
  ['pycone/src/pycone.cpp'],
  install: true,
  subdir: 'pycone',
  include_directories: includes,
  dependencies: [pybind11_dep]
)

py.extension_module(
  'gsl',
  ['pycone/src/gsl.cpp'],
  install: true,
  subdir: 'pycone',
  include_directories: includes,
  dependencies: [pybind11_dep, gsl_dep]
)
