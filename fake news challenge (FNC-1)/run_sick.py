# -*- coding: utf-8 -*-
print 'started....'
#import skipthoughts
#import eval_sick
import eval_fnc
import warnings

warnings.simplefilter("ignore")

i=4997

#model = skipthoughts.load_model()
#encoder = skipthoughts.Encoder(model)

#reload(eval_sick)
#eval_sick.evaluate(encoder, evaltest=True)

reload(eval_fnc)
eval_fnc.evaluate(encoder=None, evaltest=False)

