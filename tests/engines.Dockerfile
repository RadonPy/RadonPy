FROM python:3.12.8-slim-bookworm
ARG BUILD_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build curl ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The double-precision CPU build is the numerical reference, not a benchmark.
RUN curl -fsSL https://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gz -o /tmp/gmx.tar.gz \
    && echo 'bbda056ee59390be7d58d84c13a9ec0d4e3635617adf2eb747034922cba1f029  /tmp/gmx.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/gmx.tar.gz -C /tmp \
    && cmake -S /tmp/gromacs-2024.3 -B /tmp/gmx-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/gromacs \
       -DGMX_DOUBLE=ON -DGMX_GPU=OFF -DGMX_FFT_LIBRARY=fftpack \
       -DGMX_SIMD=SSE2 -DBUILD_TESTING=OFF -DGMX_BUILD_UNITTESTS=OFF \
       -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF -DGMX_USE_COLVARS=NONE \
    && cmake --build /tmp/gmx-build -j ${BUILD_JOBS} \
    && cmake --install /tmp/gmx-build \
    && rm -rf /tmp/gmx.tar.gz /tmp/gromacs-2024.3 /tmp/gmx-build

RUN git clone --depth 1 --branch stable_29Aug2024_update2 https://github.com/lammps/lammps.git /tmp/lammps \
    && git -C /tmp/lammps rev-parse HEAD > /opt/lammps-source-commit \
    && cmake -S /tmp/lammps/cmake -B /tmp/lmp-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/lammps \
       -DBUILD_MPI=OFF -DBUILD_OMP=ON -DPKG_MOLECULE=ON -DPKG_KSPACE=ON \
       -DPKG_RIGID=ON -DPKG_EXTRA-MOLECULE=ON -DPKG_EXTRA-DUMP=ON \
       -DPKG_EXTRA-FIX=ON -DPKG_OPENMP=ON \
    && cmake --build /tmp/lmp-build -j ${BUILD_JOBS} \
    && cmake --install /tmp/lmp-build \
    && rm -rf /tmp/lammps /tmp/lmp-build

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock
ENV GROMACS_EXEC=/opt/gromacs/bin/gmx_d LAMMPS_EXEC=/opt/lammps/bin/lmp
ENV MPLBACKEND=Agg OMP_NUM_THREADS=1
WORKDIR /work
CMD ["python", "-m", "pytest", "-c", "tests/pytest.ini", "tests", "--require-engines", "-q"]
