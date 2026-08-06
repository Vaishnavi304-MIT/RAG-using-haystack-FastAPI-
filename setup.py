from setuptools import find_packages, setup

setup(
    name="QAsystem with haystack",
    version="0.0.1",
    author="vaishnavi",
    author_email="shindevaishnavi304@gmail.com",
    packages=find_packages(),
    install_requires=["pinecone-haystack","haystack-ai","fastapi","uvicorn","python-dotenv","pathlib","sentence-transformers-haystack", "pypdf","nltk==3.9.1","huggingface-api-haystack"],
)