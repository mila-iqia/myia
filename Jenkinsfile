node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: './ci-install.sh --gpu'
  }
  try {
    stage ('Test') {
      sh script: '$HOME/miniconda/bin/conda activate test && conda env list && pytest --cov=./ --cov-report= --gpu --junit-xml test-report.xml'
    }
  } finally {
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    sh script: """
$HOME/miniconda/bin/conda activate test &&
./cov.sh &&
coverage xml
"""
    sh script: '$HOME/miniconda/bin/pip install codecov'
    withCredentials([string(credentialsId: 'myia_codecov', variable: 'CODECOV_TOKEN')]) {
      sh script: '$HOME/miniconda/bin/codecov'
    }
  }
}
