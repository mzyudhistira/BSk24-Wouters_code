The workflow for this example is the following:

   1. Run simple_massmodel.py: 
        - this trains a neural network on the AME20 data
        - ... and saves the model weight in a dedicated folder

   2. Run plot.py:
        - to plot the training and validation loss as a function of epoch
        - ... which is saved as loss.pdf
        - ... which serves as a verification step
        
   3. Run generate_mass_tables.py:
        - to ask for the predictions of a given model
        - ... which get saved in the mass_tables directory
        
   4. Analyse your results!
