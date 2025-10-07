import os

import pandas as pd
from pyteomics import mgf
from tqdm import tqdm


def remove_ptms(seq):
    return "".join([c for c in seq if "A" <= c <= "Z"])


def create_sub_mgf(mgf_file, cache_dir, num_spectra=1000):
    sub_spectra = []
    with mgf.read(
        mgf_file,
        use_index=False,
        convert_arrays=0,
        read_charges=False,
        read_ions=False,
    ) as massivekb:
        for i, spectrum in enumerate(massivekb):
            if i == num_spectra:
                break
            sub_spectra.append(spectrum)
        sub_file = os.path.join(cache_dir, f"sub_{num_spectra}.mgf")
        mgf.write(sub_spectra, sub_file)
    return sub_file


def create_sequence_index(mgf_file, cache_dir=None):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, "sequence_index.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

    with mgf.read(
        mgf_file,
        use_index=False,
        convert_arrays=0,
        read_charges=False,
        read_ions=False,
    ) as massivekb:
        sequences = []
        unmodified_sequences = []
        rts = []
        titles = []
        charges = []
        masses = []
        indexes = []
        for i, spectrum in enumerate(tqdm(massivekb, total=None)):
            if spectrum is None:
                continue

            indexes.append(i)
            sequence = spectrum["params"]["seq"]
            sequences.append(sequence)
            unmodified_sequences.append(remove_ptms(sequence))
            rts.append(spectrum["params"]["rtinseconds"])
            titles.append(spectrum["params"]["title"])
            charges.append(spectrum["params"]["charge"])
            masses.append(spectrum["params"]["pepmass"])

        df = pd.DataFrame(
            {
                "mgf_i": indexes,
                "sequence": sequences,
                "unmodified_sequence": unmodified_sequences,
                "title": titles,
                "charge": charges,
                "mass": masses,
                "rt": rts,
            }
        )

    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, "sequence_index.csv")
        df.to_csv(cache_file, index=False)

    return df
