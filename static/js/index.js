function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button ? button.querySelector('.copy-text') : null;

  if (!bibtexElement || !button || !copyText) {
    return;
  }

  const setCopied = () => {
    button.classList.add('copied');
    copyText.textContent = 'Copied';
    setTimeout(() => {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  };

  if (navigator.clipboard) {
    navigator.clipboard.writeText(bibtexElement.textContent).then(setCopied).catch(() => {
      fallbackCopy(bibtexElement.textContent);
      setCopied();
    });
    return;
  }

  fallbackCopy(bibtexElement.textContent);
  setCopied();
}

function fallbackCopy(text) {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.setAttribute('readonly', '');
  textArea.style.position = 'absolute';
  textArea.style.left = '-9999px';
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand('copy');
  document.body.removeChild(textArea);
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
}

window.addEventListener('scroll', () => {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (!scrollButton) {
    return;
  }

  if (window.pageYOffset > 300) {
    scrollButton.classList.add('visible');
  } else {
    scrollButton.classList.remove('visible');
  }
});
