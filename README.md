# Return Capping: Sample Efficient CVaR Policy Gradient Optimisation

## Installation
The easiest method of installation is to use the `$ make build` and then `$ make run` to set up a docker environment

## Running Experiments
There are five available environments
  - betting_game
  - autonomous_vehicle
  - guarded_maze
  - guarded_maze_cesor
  - lunar_lander

An example of how to run code 

`$ python main.py env=betting_game`

For Standared PPO CVaR, use the policy.alpha_batch flag to set the CVaR alpha value

`$ python main.py env=betting_game policy.alpha_batch=0.2`

For Return Capping, use the cap_return, cap_alpha and cap_tau tags. The latter two flags set the CVaR alpha and the Cap Step size respectively 

`$ python main.py env=betting_game cap_return=True cap_alpha=0.2 cap_tau=0.1`

To set minimum cap value, use the initial_return_cap and minimum_return_cap flags

`$ python main.py env=betting_game cap_return=True cap_alpha=0.2 cap_tau=0.1 initial_return_cap=-0.1 minimum_return_cap=-0.1`
