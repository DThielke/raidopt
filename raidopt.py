import pandas as pd
import numpy as np
import xlwings as xw
import cvxpy as cp
import random

# Read roster and extract character data
roster_data = xw.sheets["LootNeed"].range("InputAnchor").expand().options(pd.DataFrame, index=False, header=True).value
n_roster = roster_data.shape[0]
n_bosses = roster_data.shape[1] - 4
boss_names = roster_data.columns[-n_bosses:]
characters = roster_data["Character"]
is_tank = roster_data["Role"] == "1-Tank"
is_healer = roster_data["Role"] == "2-Healer"
is_melee = roster_data["Role"] == "3-mDPS"
has_immunity = roster_data["Class"].isin(["Paladin", "Demon Hunter", "Rogue", "Hunter", "Mage"])

# Parse boss preferences and rescale so higher values => more preference
boss_preferences = roster_data.iloc[:, -n_bosses:]
adj_boss_preferences = boss_preferences.replace("P", n_bosses + 1)  # don't care if in
adj_boss_preferences = adj_boss_preferences.replace("O", n_bosses + 101)  # really don't want to be in
adj_boss_preferences = adj_boss_preferences.astype("int")
adj_boss_preferences = (n_bosses + 1) - adj_boss_preferences

# Read general options
general_options = xw.sheets["Constraints"].range("GeneralOptionsAnchor").expand().options(dict).value
min_bosses_for_vault = int(general_options["Min Bosses for Vault"])
randomize_preference_below_vault_cutoff = general_options["Randomize Preferences Below Vault Cutoff"]

# Read boss constraints
constraint_df = (
    xw.sheets["Constraints"]
    .range("ConstraintsAnchor")
    .expand()
    .options(pd.DataFrame, index=True, header=True)
    .value.astype(int)
    .T
)
n_raiders = constraint_df["Raiders"]
n_tanks = constraint_df["Tanks"]
n_healers = constraint_df["Healers"]
min_melee = constraint_df["Min Melee"]
max_melee = constraint_df["Max Melee"]
n_immunities = constraint_df["Immunities"]

# Read fixed comp inputs
fixed_comp_df = (
    xw.sheets["Constraints"]
    .range("FixedCompAnchor")
    .expand()
    .options(pd.DataFrame, index=False, header=True)
    .value.iloc[:, -n_bosses:]
)
min_comp = fixed_comp_df.fillna(0).astype("int")
max_comp = fixed_comp_df.fillna(1).astype("int")
# If we're forcing the player to be out for too many bosses, we need to relax the vault constraint
min_bosses_per_player = np.minimum(min_bosses_for_vault, n_bosses - (max_comp == 0).sum(axis=1))

# Randomize low preferences if configured
# This is done primarily to balance the fact that some people are still listing numerical preferences for all bosses
# while others are putting "P" for most bosses. Without this, the people with any numerical preference will be put in
# for far more bosses.
def randomize_low_preferences(preferences, low_threshold):
    """Shuffles preferences below a threshold.

    Ensures that preferences from 1 to low_threshold are included in a random order, randomly replacing 0s if needed.
    """
    preferences = np.array(preferences)
    is_low = preferences <= low_threshold
    num_nonzero = min(sum(is_low), low_threshold)
    num_zero = min(sum(is_low) - num_nonzero, sum(preferences == 0))
    random_preferences = [0] * num_zero
    if num_nonzero > 0:
        random_preferences += list(range(1, num_nonzero + 1))
    random_preferences = np.array(random_preferences)
    random.shuffle(random_preferences)
    preferences[is_low] = random_preferences
    return preferences


if randomize_preference_below_vault_cutoff:
    opt_outs = adj_boss_preferences == -100
    random_cutoff = n_bosses - min_bosses_for_vault
    adj_boss_preferences = adj_boss_preferences.apply(
        randomize_low_preferences, low_threshold=random_cutoff, axis=1, raw=True
    )
    adj_boss_preferences[opt_outs] = -100

# Square the boss preferences to make the difference between ranks more dramatic
adj_boss_preferences = (adj_boss_preferences ** 2) * np.sign(adj_boss_preferences)

# Setup the optimization problem and solve
x = cp.Variable((n_roster, n_bosses), integer=True)
objective = cp.Maximize(cp.sum(cp.multiply(x, adj_boss_preferences)))
constraints = [
    x >= min_comp,
    x <= max_comp,
    cp.sum(x, axis=0) == n_raiders.values,
    cp.sum(x, axis=1) >= min_bosses_per_player.values,
    is_tank.values @ x == n_tanks,
    is_healer.values @ x == n_healers,
    is_melee.values @ x >= min_melee,
    is_melee.values @ x <= max_melee,
    has_immunity.values @ x >= n_immunities,
]
for required_class in ["Monk", "Priest", "Demon Hunter", "Warrior", "Mage", "Warlock"]:
    is_class = roster_data["Class"] == required_class
    constraints.append(is_class.values @ x >= 1)
prob = cp.Problem(objective, constraints)
prob.solve(verbose=True)
roster_opt = pd.DataFrame(x.value, index=characters, columns=boss_names).astype("int")

# Output the optimal roster
xw.sheets["Output Comp"].range("OutputCompAnchor").options(index=False, header=True).value = roster_opt
