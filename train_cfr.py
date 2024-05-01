import rlcard
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)
from rlcard.utils import (
    tournament,
    Logger,
    plot_curve,
)

env = rlcard.make('limit-holdem',
        config={
            'allow_step_back': True,
        })
eval_env = rlcard.make('limit-holdem',)
agent = CFRAgent(env,"experiments/limit_holdem_cfr_result/cfr_model",)
eval_env.set_agents([agent,RandomAgent(num_actions=env.num_actions),])

with Logger("experiments/limit_holdem_cfr_result") as logger:
    for episode in range(1000):
        agent.train()
        print('\rIteration {}'.format(episode), end='')
        # Evaluate the performance. Play with Random agents.
        if episode % 50 == 0:
            logger.log_performance(
                env.timestep,
                tournament(
                    eval_env,
                    10000,
                )[0]
            )

    # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path

plot_curve(csv_path, fig_path, 'cfr')
agent.save()