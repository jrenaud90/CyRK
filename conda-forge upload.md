After new PyPI upload
* Install greyskull: `conda install -c conda-forge grayskull`
* grayskull pypi --strict-conda-forge YOUR_PACKAGE_NAME
* Copy over the the top part of the resuling yaml file to `meta.yaml`:
  * Make sure all of these are updated: version, source url, sha256, and any build requirement changes. 
  * Note that the `meta.yaml` file contains info not produced by grayskull, so it should not totally overwrite the current `meta.yaml`

_Note: it looks like pypi provides a sha256 already so just need a way to copy that and overwrite the yaml_

* Follow the steps here to upload the recipe to conda-forge: https://conda-forge.org/docs/maintainer/adding_pkgs/#step-by-step-instructions

TODO: Look into a way to automate this with github actions after the PyPI upload is completed during build phase.