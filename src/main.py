from flask import escape, abort
import flask
import cvxpy as cp
import numpy as np
import pandas as pd
import random
import traceback


def parse_json(json_data):
    general_options = dict(json_data["generalOptions"])

    boss_constraints = pd.DataFrame(json_data["bossConstraints"][1:], columns=json_data["bossConstraints"][0])
    boss_constraints = boss_constraints.set_index(boss_constraints.iloc[:, 0]).iloc[:, 1:]
    boss_constraints = boss_constraints.drop(columns="")
    boss_constraints = boss_constraints.rename_axis(columns="Boss")

    player_constraints = pd.DataFrame(json_data["playerConstraints"][1:], columns=json_data["playerConstraints"][0])
    player_constraints = player_constraints.set_index("Character")
    player_constraints = player_constraints.drop(columns="").drop(index="")

    player_info = player_constraints[["Status", "Role", "2nd Role", "Class", "Main"]]
    player_info.loc[player_info["Main"] == "", "Main"] = None
    player_info["Main"] = player_info["Main"].fillna(player_info.index.to_series())

    player_constraints = player_constraints.drop(columns=player_info.columns)
    player_constraints = player_constraints.replace("", np.nan)
    player_constraints = player_constraints.reindex(player_info.index)

    loot_needs = pd.DataFrame(json_data["lootNeeds"][3:], columns=json_data["lootNeeds"][0])
    loot_needs = loot_needs.set_index(loot_needs.iloc[:, 0]).iloc[:, 6:].rename_axis(index="Character")
    loot_needs = loot_needs.drop(columns="").drop(index="")
    loot_needs = loot_needs.reindex(player_info.index)
    loot_needs = loot_needs.replace("", "P").fillna("P")

    return (
        general_options,
        boss_constraints,
        player_info,
        player_constraints,
        loot_needs,
    )


def clean_loot_needs(loot_needs, options):
    n_bosses = loot_needs.shape[1]
    loot_needs = loot_needs.replace("P", n_bosses + 1)  # don't care if in
    loot_needs = loot_needs.replace("O", n_bosses + 101)  # really don't want to be in
    loot_needs = loot_needs.astype("int")
    loot_needs = (n_bosses + 1) - loot_needs
    if options["Randomize Preferences Below Vault Cutoff"]:
        opt_outs = loot_needs == -100
        threshold = n_bosses - options["Min Bosses for Vault"]
        loot_needs = loot_needs.apply(randomize_low_values, threshold=threshold, axis=1, raw=True)
        loot_needs[opt_outs] = -100
    return loot_needs


def randomize_low_values(values, threshold):
    values = np.array(values)
    is_low = values <= threshold
    num_nonzero = min(sum(is_low), threshold)
    num_zero = min(sum(is_low) - num_nonzero, sum(values <= 0))
    random_values = [0] * num_zero
    if num_nonzero > 0:
        random_values += list(range(1, num_nonzero + 1))
    random_values = np.array(random_values)
    random.shuffle(random_values)
    values[is_low] = random_values
    return values


def optimize(options, boss_constraints, player_info, player_constraints, loot_needs):
    n_players = player_info.shape[0]
    n_bosses = boss_constraints.shape[1]
    n_raiders = boss_constraints.loc["Raiders"]
    n_tanks = boss_constraints.loc["Tanks"]
    n_healers = boss_constraints.loc["Healers"]
    min_mdps = boss_constraints.loc["Min Melee"]
    max_mdps = boss_constraints.loc["Max Melee"]
    min_rdps = n_raiders - n_tanks - n_healers - max_mdps
    max_rdps = n_raiders - n_tanks - n_healers - min_mdps

    roles = player_info["Role"]
    roles2 = player_info["2nd Role"]
    classes = player_info["Class"]
    mains = player_info["Main"]
    is_tank = (roles == "1-Tank") | (roles2 == "1-Tank")
    is_healer = (roles == "2-Healer") | (roles2 == "2-Healer")
    is_mdps = (roles == "3-mDPS") | (roles2 == "3-mDPS")
    is_rdps = (roles == "4-rDPS") | (roles2 == "4-rDPS")
    pref_tank = roles == "1-Tank"
    pref_healer = roles == "2-Healer"
    pref_mdps = roles == "3-mDPS"
    pref_rdps = roles == "4-rDPS"

    # Number of bosses for players are summed across mains/alts
    main_constraints_min = player_constraints.fillna(0).groupby(mains).max()
    main_constraints_max = player_constraints.fillna(1).groupby(mains).max()
    character_constraints_min = player_constraints.fillna(0).astype("int")
    character_constraints_max = player_constraints.fillna(1).astype("int")
    # If we're forcing the player to be out for too many bosses, we need to relax the vault constraint
    min_bosses_per_player = np.minimum(
        options["Min Bosses for Vault"],
        n_bosses - (main_constraints_max == 0).sum(axis=1),
    )

    # Variables:
    x_tank = cp.Variable((n_players, n_bosses), integer=True)
    x_heal = cp.Variable((n_players, n_bosses), integer=True)
    x_mdps = cp.Variable((n_players, n_bosses), integer=True)
    x_rdps = cp.Variable((n_players, n_bosses), integer=True)
    is_in = x_tank + x_heal + x_mdps + x_rdps

    # General constraints:
    constraints = [
        x_tank >= 0,
        x_tank <= character_constraints_max.mul(is_tank, axis=0),
        x_heal >= 0,
        x_heal <= character_constraints_max.mul(is_healer, axis=0),
        x_mdps >= 0,
        x_mdps <= character_constraints_max.mul(is_mdps, axis=0),
        x_rdps >= 0,
        x_rdps <= character_constraints_max.mul(is_rdps, axis=0),
        is_in >= character_constraints_min,
        is_in <= character_constraints_max,
        cp.sum(is_in, axis=0) == n_raiders,
        cp.sum(x_tank, axis=0) == n_tanks,
        cp.sum(x_heal, axis=0) == n_healers,
        cp.sum(x_mdps, axis=0) >= min_mdps,
        cp.sum(x_mdps, axis=0) <= max_mdps,
        cp.sum(x_rdps, axis=0) >= min_rdps,
        cp.sum(x_rdps, axis=0) <= max_rdps,
    ]

    # Class buff constraints:
    for required_class in [
        "Monk",
        "Priest",
        "Demon Hunter",
        "Warrior",
        "Mage",
        "Warlock",
    ]:
        is_class = player_info["Class"] == required_class
        constraints.append(is_class.values @ is_in >= 1)
    for main in player_info["Main"].unique():
        player_bosses_incl_alts = cp.sum(is_in[player_info["Main"] == main], axis=0)
        constraints.append(player_bosses_incl_alts >= main_constraints_min.loc[main])
        constraints.append(player_bosses_incl_alts <= main_constraints_max.loc[main])
        constraints.append(cp.sum(player_bosses_incl_alts) >= min_bosses_per_player[main])

    # Custom constraints:
    has_speed_boost = classes.isin(["Druid", "Shaman"])
    constraints.append(has_speed_boost.values @ is_in >= boss_constraints.loc["Min Speed Boosts"])

    has_immunity = classes.isin(["Paladin", "Demon Hunter", "Rogue", "Hunter", "Mage"])
    constraints.append(has_immunity.values @ is_in >= boss_constraints.loc["Min Immunities"])

    is_hunter = classes == "Hunter"
    constraints.append(is_hunter.values @ is_in >= boss_constraints.loc["Min Hunters"])

    is_rogue = classes == "Rogue"
    constraints.append(is_rogue.values @ is_in >= boss_constraints.loc["Min Rogues"])

    is_boomkin = (classes == "Druid") & is_rdps
    constraints.append(is_boomkin.values @ is_in >= boss_constraints.loc["Min Boomkin"])

    # Run optimization
    squared_loot_needs = (loot_needs ** 2) * np.sign(loot_needs)
    pref_main_roles = (
        pref_tank.values @ x_tank + pref_healer.values @ x_heal + pref_mdps.values @ x_mdps + pref_rdps.values @ x_rdps
    )
    objective = cp.Maximize(cp.sum(cp.multiply(is_in, squared_loot_needs) + 0.1 * cp.sum(pref_main_roles)))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)
    if prob.status != "optimal":
        return None
    opt_roster = pd.DataFrame(is_in.value, index=player_info.index, columns=boss_constraints.columns).astype("int")
    opt_tanks = pd.DataFrame(x_tank.value, index=player_info.index, columns=boss_constraints.columns).astype("int")
    opt_heal = pd.DataFrame(x_heal.value, index=player_info.index, columns=boss_constraints.columns).astype("int")
    opt_mdps = pd.DataFrame(x_mdps.value, index=player_info.index, columns=boss_constraints.columns).astype("int")
    opt_rdps = pd.DataFrame(x_rdps.value, index=player_info.index, columns=boss_constraints.columns).astype("int")
    return opt_roster, opt_tanks, opt_heal, opt_mdps, opt_rdps


def optimize_comp(request: flask.Request):
    """
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    """
    if request.method != "POST":
        abort(405)
    json_data = request.get_json(silent=True)
    if not json_data:
        abort(404)
    try:
        (
            options,
            boss_constraints,
            player_info,
            player_constraints,
            loot_needs,
        ) = parse_json(json_data)
        loot_needs = clean_loot_needs(loot_needs, options)
        opt_results = optimize(options, boss_constraints, player_info, player_constraints, loot_needs)
        if opt_results is None:
            return "Problem is infeasible. Check the constraints.", 500
        # Ensure the output has 50 rows and 14 columns
        opt_roster, opt_tanks, opt_heal, opt_mdps, opt_rdps = opt_results
        opt_roster = opt_tanks + 2 * opt_heal + 3 * opt_mdps + 4 * opt_rdps
        opt_roster = opt_roster.reset_index().reindex(range(50)).set_index("Character").fillna(0)
        opt_roster = opt_roster.T.reset_index().reindex(range(14)).set_index("Boss").fillna(0).T
        player_info = player_info.reset_index().reindex(range(50)).set_index("Character")
        result = pd.concat([player_info, opt_roster], axis=1).reset_index().T.reset_index().T.to_json(orient="values")
    except:
        return traceback.format_exc(), 500
    return result
