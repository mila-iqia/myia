node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: './ci-install.sh --gpu'
  }
  try {
    stage ('Test') {
      sh script: """
. $HOME/miniconda/etc/profile.d/conda.sh &&
conda activate test &&
pytest --cov=./ --cov-report= --gpu --junit-xml test-report.xml
"""
    }
  } finally {
    junit 'test-report.xml'
  }
  stage ('Coverage') {
    sh script: """
. $HOME/miniconda/etc/profile.d/conda.sh &&
conda activate test &&
./cov.sh &&
coverage xml
"""
    sh script: """
. $HOME/miniconda/etc/profile.d/conda.sh &&
conda activate test &&
pip install codecov
"""
    withCredentials([string(credentialsId: 'myia_codecov', variable: 'CODECOV_TOKEN')]) {
      sh script: """
. $HOME/miniconda/etc/profile.d/conda.sh &&
conda activate test &&
codecov --commit `git rev-parse origin/$BRANCH_NAME`
"""
    }
  }
}
