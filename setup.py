import setuptools
import atexit
from setuptools.command.develop import develop
from setuptools.command.install import install

def _download_models():
    from esm import pretrained
    from transformers import T5Tokenizer, T5EncoderModel, logging
    logging.set_verbosity_error()
    _, _ = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S") 

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def __init__(self, *args, **kwargs):
        super(PostDevelopCommand, self).__init__(*args, **kwargs)
        atexit.register(_download_models)
        
class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def __init__(self, *args, **kwargs):
        super(PostInstallCommand, self).__init__(*args, **kwargs)
        atexit.register(_download_models)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup_requires = ['numpy','transformers','fair-esm','sentencepiece','torch>=1.6']

install_requires = [
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'Bio',
        'torch>=1.6',
        'onnxruntime>=1.7.0',
        'fair-esm',
        'transformers',
        'pytorch_lightning',
        'sentencepiece'
        ]

setuptools.setup(
    name="DeepLoc2",
    version="1.0.0",
    author="Jose Juan Almagro Armenteros",
    author_email="jjaa@stanford.edu",
    description="Prediction of subcellular localization",
    #scripts=['bin/deeploc2'],
    entry_points={'console_scripts':['deeploc2=DeepLoc2.deeploc2:predict']},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://services.healthtech.dtu.dk/service.php?DeepLoc-2.0",
    project_urls={
        "Bug Tracker": "https://services.healthtech.dtu.dk/service.php?DeepLoc-2.0",
    },
    install_requires=install_requires,
    setup_requires=setup_requires,
    cmdclass={'develop':PostDevelopCommand,'install':PostInstallCommand},
    packages=setuptools.find_packages(),
    package_data={'DeepLoc2': ['models/*', 'models/models_esm1b/*','models/models_prott5/*',
                  'models/models_esm1b/signaltype/*','models/models_prott5/signaltype/*']},
    python_requires=">=3.6",
)

