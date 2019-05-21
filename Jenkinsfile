node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    sh script: '.testing/install.sh'
  }
  stage ('Test') {
    sh script: '$HOME/miniconda/bin/pytest --cov=./ --gpu'
  }
  stage ('Coverage') {
    sh script: 'codecov'
  }
}
