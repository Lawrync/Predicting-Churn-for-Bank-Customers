import pickle

files = [
    'preprocessor.pkl',
    'best_model.pkl',
    'best_xgb_model.pkl',
    'scaler.pkl',
    'feature_names.pkl'
]

for f in files:
    try:
        with open(f, 'rb') as file:
            obj = pickle.load(file)
        with open(f, 'wb') as file:
            pickle.dump(obj, file, protocol=5)
        print(f'{f} re-saved successfully.')
    except Exception as e:
        print(f'Error with {f}: {e}')
