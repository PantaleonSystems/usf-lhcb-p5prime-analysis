# scripts/generate_p5p_csv.py
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Paths to the YAML files (adjust if necessary)
# ------------------------------------------------------------
YAML_DIR = Path("data/raw/HEPData-ins1409497-v1-yaml")
FL_FILE = YAML_DIR / "Table1.yaml"   # contains F_L values and errors
S5_FILE = YAML_DIR / "Table2.yaml"   # contains S_5 values and errors

def load_values(yaml_file, observable_name):
    """Extract values and symmetric errors (stat+syst in quadrature) for a given observable."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Find the dependent variable block with the right name
    target = None
    for dep in data['dependent_variables']:
        if observable_name in dep['header']['name']:
            target = dep
            break
    if target is None:
        raise ValueError(f"Observable {observable_name} not found in {yaml_file}")
    
    vals = []
    errs = []
    for v in target['values']:
        val = v['value']
        # Quadratic sum of all errors (stat and syst)
        err2 = 0.0
        for e in v['errors']:
            if 'symerror' in e:
                err2 += e['symerror']**2
            elif 'asymerror' in e:
                # use the average of plus and minus as symmetric approximation
                plus = e['asymerror']['plus']
                minus = e['asymerror']['minus']
                err2 += ((plus - minus)/2)**2
        vals.append(val)
        errs.append(np.sqrt(err2))
    return vals, errs

def main():
    # Load F_L and S_5 for the appropriate bins
    fl_vals, fl_errs = load_values(FL_FILE, "$F_{\\rm L}$")
    s5_vals, s5_errs = load_values(S5_FILE, "$S_{5}$")
    
    # The independent variable (q² bins) is common; we extract from FL_FILE
    with open(FL_FILE, 'r') as f:
        data = yaml.safe_load(f)
    indep = data['independent_variables'][0]
    bins = []
    for v in indep['values']:
        low = v['low']
        high = v['high']
        bins.append((low + high)/2.0)
    
    # We only keep the seven bins that were used in the analysis
    # (the first bin q² < 1 was excluded)
    # Alternatively, you can keep all and filter later.
    # Here we assume the YAMLs already contain the bins in the same order as the CSV.
    # In your final CSV you have 7 bins: 1.8, 3.25, 5.0, 7.0, 11.75, 16.0, 18.0
    # We will filter to match those centers (tolerance 0.1)
    target_centers = np.array([1.8, 3.25, 5.0, 7.0, 11.75, 16.0, 18.0])
    
    selected = []
    for i, cen in enumerate(bins):
        if np.any(np.abs(target_centers - cen) < 0.1):
            selected.append(i)
    
    q2_out = [bins[i] for i in selected]
    fl_out = [fl_vals[i] for i in selected]
    fl_err_out = [fl_errs[i] for i in selected]
    s5_out = [s5_vals[i] for i in selected]
    s5_err_out = [s5_errs[i] for i in selected]
    
    # Compute P5' and propagate errors (uncorrelated, diagonal)
    p5_vals = []
    p5_errs = []
    for fl, fl_e, s5, s5_e in zip(fl_out, fl_err_out, s5_out, s5_err_out):
        denom = np.sqrt(fl * (1 - fl))
        if denom == 0:
            p5_vals.append(np.nan)
            p5_errs.append(np.nan)
            continue
        p5 = s5 / denom
        dP_dS5 = 1.0 / denom
        dP_dFL = -0.5 * s5 * (1 - 2*fl) / (denom**3)
        err_p5 = np.sqrt((dP_dS5 * s5_e)**2 + (dP_dFL * fl_e)**2)
        p5_vals.append(p5)
        p5_errs.append(err_p5)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame({
        'q2_center': q2_out,
        'value': p5_vals,
        'error': p5_errs
    })
    df.to_csv("data/p5p_observables.csv", index=False)
    print("Generated data/p5p_observables.csv with", len(df), "bins.")
    print(df)

if __name__ == "__main__":
    main()