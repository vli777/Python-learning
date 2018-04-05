rom sklearn.hmm import GaussianHMM
import numpy as np

# Define batch transform which will be called
# periodically with a continuously updated array
# of the most recent trade data.
@batch_transform(refresh_period=100, window_length=300)
def HMM(data, means_prior=None):
    # data is _not_ an event-frame, but an array
    # of the most recent trade events

    # Create scikit-learn model using the means
    # from the previous model as a prior
    model = GaussianHMM(3,
                        covariance_type="diag",
                        n_iter=10,
                        means_prior=means_prior,
                        means_weight=0.5)

    # Extract variation and volume
    diff = data.variation[3951].values
    volume = data.volume[3951].values
    X = np.column_stack([diff, volume])

    # Estimate model
    model.fit([X])

    return model

# Put any initialization logic here  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    # some sids to look at
    context.sid = sid(3951)
    context.means_prior = None

def handle_data(context, data):
    c = context

    # add the day's price range to the list for this sid
    data[c.sid]['variation'] = (data[c.sid].close_price - data[c.sid].open_price)

    # Pass event frame to batch_transform
    # Will _not_ directly call the transform but append
    # data to a window until full and then compute.
    model = HMM(data, means_prior=c.means_prior)

    if model is None:
        return

    # Remember mean for the prior
    c.means_prior = model.means_

    data_vec = [data[c.sid].variation, data[c.sid].volume]
    state = model.predict([data_vec])

    log.info(state)
