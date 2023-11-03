import setuptools
import atexit
from setuptools.command.develop import develop
from setuptools.command.install import install

def _download_models():
    from esm import pretrained
    logging.set_verbosity_error()
    _, _ = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D") 

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

setup_requires = ['numpy','fair-esm','torch>=1.6']

install_requires = [
        'numpy',
        'matplotlib',
        'pandas',
        'Bio',
        'torch>=1.6',
        'fair-esm',
        ]

setuptools.setup(
    name="DeepLocPro",
    version="1.0.0",
    author="Jaime Moreno",
    author_email="jxmm@novonordisk.com",
    description="Prediction of protein subcellular localization of Prokaryotic organisms",
    entry_points={'console_scripts':['deeplocpro=DeepLocPro.deeplocpro:predict']},
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
    package_data={'DeepLocPro': ['models/*', 'models/checkpoints/*']},
    python_requires=">=3.6",
)

