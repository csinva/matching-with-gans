# labels for annotations

Values in annotation dict correspond to indices in these lists (e.g. hair length value 0 = 'Very short')

```python
annotation_dict_names {
    'hair-length': ['Very short', 'Short', 'Medium', 'Long', 'Very long'],
    'ethnicity': ['East Asian', 'South Asian', 'African', 'Latino', 'Middle East', 'Caucasian'],
    'age': ['Child', 'Teen', 'Young adult', 'Adult', 'Middle age', 'Senior'],
    'gender': ['Female', 'Probably female', 'In between', 'Probably male','Male'],
    'skin-color': ['Light', 'Fair', 'Medium', 'Olive', 'Brown', 'Black'],
    'makeup': ['None', 'Minimal', 'Full', 'Showy'],
    'facial-hair': ['None', 'Minimal', 'Mustache', 'Beard', 'Full']
 }
```

`annotations.pkl` contains a dictionary of annotations for 5000 random images generated from the stylegan2 latent space. The keys are the same as the dict above and the values are each a numpy array of size (5000, 7), where 7 is the number of annotators.