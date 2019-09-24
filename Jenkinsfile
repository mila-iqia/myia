node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: './ci-install.sh --gpu'
  }
  try {
    stage ('Test') {
      sh script: './cov.sh main $HOME/miniconda/bin/pytest --cov=./ --cov-report= --gpu --junit-xml test-report.xml'
      sh script: 'MYIA_BACKEND=relay ./cov.sh relay $HOME/miniconda/bin/pytest --cov=./ tests/test_api.py tests/test_compile.py --deselect=tests/test_compile.py::test_array_setitem tests/test_grad.py tests/test_model.py --cov-report= --gpu'
    }
  } finally {
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    sh script: '$HOME/miniconda/bin/pip install codecov'
    sh script: '$HOME/miniconda/bin/coverage combine *.coverage'
    sh script: '$HOME/miniconda/bin/coverage report -m'
    sh script: '$HOME/miniconda/bin/coverage xml'
    withCredentials([string(credentialsId: 'myia_codecov', variable: 'CODECOV_TOKEN')]) {
      sh script: '$HOME/miniconda/bin/codecov'
    }
  }
}
