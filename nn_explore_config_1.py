loadstuff = False
filename = 'tobias_test_tools'
path = ['emg_data','ipa_emg']
output_classes = 2
activation = 'relu'

dic = { 1: 'Tasse aufnehmen',
        2: 'Tasse halten',
        3: 'Tasse abstellen',
        4: 'Tasse hoch&runter',
        8: 'Ruhe (Supination)',
        9: 'Ruhe (Pronation)'
      }
#its = ['cupv1','cupv2','kettlev1','data','loosev1','jascha','markus','Sirius', 'korbi','tobias_1','tobias_2']
its = ['tobias','tobias2','tobias3','tobias4Tool']

explorationStruct = { 'hidden_layer':      [1,2,1],
                      'neurons_per_layer': [75,200,25],
                      'filter_layer':      [50,150,25]
                      }

model_params = { 'model':'CNN' # or CNN

                }

data_arch = {   0: (7,8,9),
                1: (2,4),
             }

convFilterDict = { "conv":  {
                            "filters":       30,
                            "kernel_size":   (1,6),
                            "input_shape":   (1,10,1)
                            },
                   "maxpool":{"pool_size":(1,5)}
                 }
