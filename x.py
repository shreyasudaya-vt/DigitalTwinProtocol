import pandas as pd
df = pd.read_csv("telemetry_Scenario_C.csv")
clean = df[(df['Time'] >= 15.0) & (df['Time'] < 30.0)]

print(clean['Hamming_Distance'].describe())
print(f"\nHD > 8 rate: {(clean['Hamming_Distance'] > 8).mean()*100:.1f}%")
print(f"Spatial alarm rate: {clean['Spatial_Alarm'].mean()*100:.1f}%")
print(f"Innovation > threshold rate: {(clean['Innovation'].abs() > clean['Dynamic_Threshold']).mean()*100:.1f}%")
