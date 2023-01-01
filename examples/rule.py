import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np

front_ang = ast_ang - ship_ang
if abs(front_ang < 15):
    ownship.shot()


def closest_rule(ship_position, ast_position):
    A1 = ctrl.Antecedent(np.arange(0, 301, 1), 'A1')
    A1['near'] = fuzz.trimf(A1.universe, [-100, 0, 100])
    A1['close'] = fuzz.trimf(A1.universe, [0, 100, 200])
    A1['far'] = fuzz.trimf(A1.universe, [100, 200, 300])

    B1 = ctrl.Antecedent(np.arange(0, 361), 'B1')
    B1['front'] = fuzz.trimf(B1.universe, [-30, 0, 30])
    B1['middle'] = fuzz.trimf(B1.universe, [0,60, 120])
    B1['back'] = fuzz.trimf(B1.universe, [60, 210, 360])


    Con1 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con1')
    Con1['Very Low'] = fuzz.trimf(Con1.universe, [-25, 0, 25])
    Con1['Low'] = fuzz.trimf(Con1.universe, [0, 25, 50])
    Con1['Medium'] = fuzz.trimf(Con1.universe, [25, 50, 75])
    Con1['High'] = fuzz.trimf(Con1.universe, [50, 75, 100])
    Con1['Very High'] = fuzz.trimf(Con1.universe, [75, 100, 125])

    Con1_rule1 = ctrl.Rule(antecedent=(A1['near'] & B1['front']),
                                consequent=Con1['Very High'], label='Con1_rule1')
    Con1_rule2 = ctrl.Rule(antecedent=(A1['close'] & B1['front']),
                                consequent=Con1['High'], label='Con1_rule2')
    Con1_rule3 = ctrl.Rule(antecedent=(A1['far'] & B1['front']),
                                consequent=Con1['Medium'], label='Con1_rule3')
    Con1_rule4 = ctrl.Rule(antecedent=(A1['near'] & B1['middle']),
                                consequent=Con1['Very High'], label='Con1_rule4')
    Con1_rule5 = ctrl.Rule(antecedent=(A1['close'] & B1['middle']),
                                consequent=Con1['High'], label='Con1_rule5')
    Con1_rule6 = ctrl.Rule(antecedent=(A1['far'] & B1['middle']),
                                consequent=Con1['Medium'], label='Con1_rule6')
    Con1_rule7 = ctrl.Rule(antecedent=(A1['near'] & B1['back']),
                                consequent=Con1['Very High'], label='Con1_rule7')
    Con1_rule8 = ctrl.Rule(antecedent=(A1['close'] & B1['back']),
                                consequent=Con1['High'], label='Con1_rule8')
    Con1_rule9 = ctrl.Rule(antecedent=(A1['far'] & B1['back']),
                                consequent=Con1['Medium'], label='Con1_rule9')

    Con1_system = ctrl.Con1trolSystem(rules=[Con1_rule1, Con1_rule2, Con1_rule3, Con1_rule4,
                                                 Con1_rule5, Con1_rule6, Con1_rule7, Con1_rule8,
                                                 Con1_rule9])
    self.Con1_sim = ctrl.Con1trolSystemSimulation(Con1_system)

    A2 = ctrl.Antecedent(np.arange(0, 361), 'A2')
    A2['near'] = fuzz.trimf(A2.universe, [-15, 0, 15])
    A2['close'] = fuzz.trimf(A2.universe, [15, 45, 75])
    A2['far'] = fuzz.trimf(A2.universe, [75, 180, 285])
    # shortest_distance < 50 + (12 * clast_size)
    B2= ctrl.Antecedent(np.arange(0, 361), 'B2')
    B2['front'] = fuzz.trimf(B1.universe, [-30, 0, 30])
    B2['middle'] = fuzz.trimf(B1.universe, [0, 60, 120])
    B2['back'] = fuzz.trimf(B1.universe, [60, 210, 360])



    Con2 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con2')
    Con2['Very Low'] = fuzz.trimf(Con2.universe, [-25, 0, 25])
    Con2['Low'] = fuzz.trimf(Con2.universe, [0, 25, 50])
    Con2['Medium'] = fuzz.trimf(Con2.universe, [25, 50, 75])
    Con2['High'] = fuzz.trimf(Con2.universe, [50, 75, 100])
    Con2['Very High'] = fuzz.trimf(Con2.universe, [75, 100, 125])
    Con2_rule1 = ctrl.Rule(antecedent=(A2['near'] & B2['In Sights']),
                                consequent=Con2['Very High'], label='Con2_rule1')
    Con2_rule2 = ctrl.Rule(antecedent=(A2['Close'] & B2['In Sights']),
                                consequent=Con2['High'], label='Con2_rule2')
    Con2_rule3 = ctrl.Rule(antecedent=(A2['Far'] & B2['In Sights']),
                                consequent=Con2['Medium'], label='Con2_rule3')
    Con2_rule4 = ctrl.Rule(antecedent=(A2['Imminent'] & B2['Close']),
                                consequent=Con2['Very High'], label='Con2_rule4')
    Con2_rule5 = ctrl.Rule(antecedent=(A2['Close'] & B2['Close']),
                                consequent=Con2['High'], label='Con2_rule5')
    Con2_rule6 = ctrl.Rule(antecedent=(A2['Far'] & B2['Close']),
                                consequent=Con2['Medium'], label='Con2_rule6')
    Con2_rule7 = ctrl.Rule(antecedent=(A2['Imminent'] & B2['Far']),
                                consequent=Con2['Very High'], label='Con2_rule7')
    Con2_rule8 = ctrl.Rule(antecedent=(A2['Close'] & B2['Far']),
                                consequent=Con2['High'], label='Con2_rule8')
    Con2_rule9 = ctrl.Rule(antecedent=(A2['Far'] & B2['Far']),
                                consequent=Con2['Medium'], label='Con2_rule9')
    Con2_system = ctrl.ControlSystem(rules=[Con2_rule1, Con2_rule2, Con2_rule3, Con2_rule4,
                                                 Con2_rule5, Con2_rule6, Con2_rule7, Con2_rule8,
                                                 Con2_rule9])
    self.Con2_sim = ctrl.ControlSystemSimulation(Con2_system)



    f_orientation_size3 = ctrl.Antecedent(np.arange(-91, 271, 1), 'f_orientation_size3')
    f_orientation_size3['In Sights'] = fuzz.trimf(f_orientation_size3.universe, [-15, 0, 15])
    f_orientation_size3['Close'] = fuzz.trimf(f_orientation_size3.universe, [15, 45, 75])
    f_orientation_size3['Far'] = fuzz.trimf(f_orientation_size3.universe, [75, 180, 285])
    f_hypotenuse_size3 = ctrl.Antecedent(np.arange(0, self.roe_zone + 1, 1), 'f_hypotenuse_size3')
    f_hypotenuse_size3['Imminent'] = fuzz.trimf(f_hypotenuse_size3.universe, [-80, 0, 80])
    f_hypotenuse_size3['Close'] = fuzz.trimf(f_hypotenuse_size3.universe, [80, 140, 200])
    f_hypotenuse_size3['Far'] = fuzz.trimf(f_hypotenuse_size3.universe, [140, 200, 260])

    Target_F3 = ctrl.Consequent(np.arange(-26, 126, 1), 'Target_F3')
    Target_F3['Very Low'] = fuzz.trimf(Target_F3.universe, [-25, 0, 25])
    Target_F3['Low'] = fuzz.trimf(Target_F3.universe, [0, 25, 50])
    Target_F3['Medium'] = fuzz.trimf(Target_F3.universe, [25, 50, 75])
    Target_F3['High'] = fuzz.trimf(Target_F3.universe, [50, 75, 100])
    Target_F3['Very High'] = fuzz.trimf(Target_F3.universe, [75, 100, 125])

    Target_F3_rule1 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Imminent'] & f_orientation_size3['In Sights']),
                                consequent=Target_F3['Very High'], label='Target_F3_rule1')

    Target_F3_rule2 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Close'] & f_orientation_size3['In Sights']),
                                consequent=Target_F3['High'], label='Target_F3_rule2')

    Target_F3_rule3 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Far'] & f_orientation_size3['In Sights']),
                                consequent=Target_F3['Medium'], label='Target_F3_rule3')

    Target_F3_rule4 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Imminent'] & f_orientation_size3['Close']),
                                consequent=Target_F3['Very High'], label='Target_F3_rule4')

    Target_F3_rule5 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Close'] & f_orientation_size3['Close']),
                                consequent=Target_F3['High'], label='Target_F3_rule5')

    Target_F3_rule6 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Far'] & f_orientation_size3['Close']),
                                consequent=Target_F3['Medium'], label='Target_F3_rule6')

    Target_F3_rule7 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Imminent'] & f_orientation_size3['Far']),
                                consequent=Target_F3['Very High'], label='Target_F3_rule7')

    Target_F3_rule8 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Close'] & f_orientation_size3['Far']),
                                consequent=Target_F3['High'], label='Target_F3_rule8')

    Target_F3_rule9 = ctrl.Rule(antecedent=(f_hypotenuse_size3['Far'] & f_orientation_size3['Far']),
                                consequent=Target_F3['Medium'], label='Target_F3_rule9')

    Target_F3_system = ctrl.ControlSystem(rules=[Target_F3_rule1, Target_F3_rule2, Target_F3_rule3, Target_F3_rule4,
                                                 Target_F3_rule5, Target_F3_rule6, Target_F3_rule7, Target_F3_rule8,
                                                 Target_F3_rule9])
    self.Target_F3_sim = ctrl.ControlSystemSimulation(Target_F3_system)

    f_orientation_size4 = ctrl.Antecedent(np.arange(-91, 271, 1), 'f_orientation_size4')
    f_orientation_size4['In Sights'] = fuzz.trimf(f_orientation_size4.universe, [-15, 0, 15])
    f_orientation_size4['Close'] = fuzz.trimf(f_orientation_size4.universe, [15, 45, 75])
    f_orientation_size4['Far'] = fuzz.trimf(f_orientation_size4.universe, [75, 180, 285])
    # shortest_distance < 50 + (12 * clast_size) We may wanna change the hypotenuse membership functions
    f_hypotenuse_size4 = ctrl.Antecedent(np.arange(0, self.roe_zone + 1, 1), 'f_hypotenuse_size4')

    f_hypotenuse_size4['Imminent'] = fuzz.trimf(f_hypotenuse_size4.universe, [-80, 0, 80])
    f_hypotenuse_size4['Close'] = fuzz.trimf(f_hypotenuse_size4.universe, [80, 140, 200])
    f_hypotenuse_size4['Far'] = fuzz.trimf(f_hypotenuse_size4.universe, [140, 200, 260])

    Target_F4 = ctrl.Consequent(np.arange(-26, 126, 1), 'Target_F4')
    Target_F4['Very Low'] = fuzz.trimf(Target_F4.universe, [-25, 0, 25])
    Target_F4['Low'] = fuzz.trimf(Target_F4.universe, [0, 25, 50])
    Target_F4['Medium'] = fuzz.trimf(Target_F4.universe, [25, 50, 75])
    Target_F4['High'] = fuzz.trimf(Target_F4.universe, [50, 75, 100])
    Target_F4['Very High'] = fuzz.trimf(Target_F4.universe, [75, 100, 125])

    Target_F4_rule1 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Imminent'] & f_orientation_size4['In Sights']),
                                consequent=Target_F4['Very High'], label='Target_F4_rule1')

    Target_F4_rule2 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Close'] & f_orientation_size4['In Sights']),
                                consequent=Target_F4['High'], label='Target_F4_rule2')

    Target_F4_rule3 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Far'] & f_orientation_size4['In Sights']),
                                consequent=Target_F4['Very Low'], label='Target_F4_rule3')

    Target_F4_rule4 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Imminent'] & f_orientation_size4['Close']),
                                consequent=Target_F4['High'], label='Target_F4_rule4')

    Target_F4_rule5 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Close'] & f_orientation_size4['Close']),
                                consequent=Target_F4['Medium'], label='Target_F4_rule5')

    Target_F4_rule6 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Far'] & f_orientation_size4['Close']),
                                consequent=Target_F4['Very Low'], label='Target_F4_rule6')

    Target_F4_rule7 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Imminent'] & f_orientation_size4['Far']),
                                consequent=Target_F4['High'], label='Target_F4_rule7')

    Target_F4_rule8 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Close'] & f_orientation_size4['Far']),
                                consequent=Target_F4['Low'], label='Target_F4_rule8')

    Target_F4_rule9 = ctrl.Rule(antecedent=(f_hypotenuse_size4['Far'] & f_orientation_size4['Far']),
                                consequent=Target_F4['Very Low'], label='Target_F4_rule9')

    Target_F4_system = ctrl.ControlSystem(rules=[Target_F4_rule1, Target_F4_rule2, Target_F4_rule3, Target_F4_rule4,
                                                 Target_F4_rule5, Target_F4_rule6, Target_F4_rule7, Target_F4_rule8,
                                                 Target_F4_rule9])
    self.Target_F4_sim = ctrl.ControlSystemSimulation(Target_F4_system)