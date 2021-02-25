#
# Copyright (C) 2019 James Parkhurst
#
# This code is distributed under the GPLv3 license.
#
from skbuild import setup


def main():
    """
    Setup the package

    """
    tests_require = ["pytest", "pytest-cov", "mock"]

    setup(
        package_dir={"guanaco": "src"},
        packages=["guanaco"],
        install_requires=["mrcfile", "numpy"],
        setup_requires=["pytest-runner"],
        tests_require=tests_require,
        test_suite="test",
        entry_points={
            "console_scripts": [
                "guanaco=guanaco.command_line:main",
                "guanaco.plot_ctf=guanaco.command_line:plot_ctf",
                "guanaco.generate_ctf=guanaco.command_line:generate_ctf",
            ]
        },
        extras_require={
            "build_sphinx": ["sphinx", "sphinx_rtd_theme"],
            "test": tests_require,
        },
    )


if __name__ == "__main__":
    main()
