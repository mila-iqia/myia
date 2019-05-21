node ('gpu') {
  stage ('Checkout') {
    checkout scm
  }
  stage ('Install') {
    cache(caches: [[$class: 'ArbitraryFileCache', excludes: '', includes: '**/*', path: '/home/jenkins/miniconda']], maxCacheSize: 3000) {
      sh script: '.testing/install.sh --gpu'
    }
  }
  stage ('Test') {
    sh script: '$HOME/miniconda/bin/pytest --cov=./ --gpu'
  }
  stage ('Coverage') {
    sh script: 'curl -s https://codecov.io/bash | bash'
  }
}
