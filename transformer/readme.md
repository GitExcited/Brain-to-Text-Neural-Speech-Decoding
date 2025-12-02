change kaggledirectory to point to the directory where each day is (inside hdf5_data_final)
You can change maxDays = 1000 to say 4 to only load 4 days into the model, I put 1000 to load them all.

In the NeuralEncoder, NeuralDecoder, NeuralToPhonemeTransformer init params:
    - can change d_model from 256 to something lower or higher as long as it divides 512 (our # of features)
    - can also change num_heads -> how many heads in multi-level attention
    - can change num_layers


In dataLoader, anything above 16 batches takes too much vram on your gpu for some reason, could be fixed.

Validation cell is computes PER and Val loss, returns a dictionary

Training cell is a basic loop across all the batches:
!!!IMPORTANT!!! the train loss, val loss, and PER is ONLY saved EVERY 40 BATCH

more info:
1 input to the model: a full trial -> one sequence (variable between 400 - 1500 shape vectors)
outputs are also variable

!!IMPORTANT!! This transformer is extremely sensitive to the learning rate, it can just not work if too high or too low
maybe try Xavier init, warmup etc...