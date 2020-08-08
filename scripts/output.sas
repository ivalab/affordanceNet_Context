begin_version
3
end_version
begin_metric
0
end_metric
4
begin_variable
var0
-1
3
Atom carry(arm, spoon)
Atom carry(arm, trowel)
Atom free(arm)
end_variable
begin_variable
var1
-1
2
Atom graspable(spoon)
NegatedAtom graspable(spoon)
end_variable
begin_variable
var2
-1
2
Atom graspable(trowel)
NegatedAtom graspable(trowel)
end_variable
begin_variable
var3
-1
2
Atom scooped(bowl, spoon)
NegatedAtom scooped(bowl, spoon)
end_variable
2
begin_mutex_group
2
0 0
1 0
end_mutex_group
begin_mutex_group
2
0 1
2 0
end_mutex_group
begin_state
2
0
0
1
end_state
begin_goal
1
3 0
end_goal
5
begin_operator
dropoff spoon arm spoon
0
2
0 0 0 2
0 1 -1 0
1
end_operator
begin_operator
dropoff trowel arm spoon
0
2
0 0 1 2
0 2 -1 0
1
end_operator
begin_operator
pickup spoon arm
0
2
0 0 2 0
0 1 0 1
1
end_operator
begin_operator
pickup trowel arm
0
2
0 0 2 1
0 2 0 1
1
end_operator
begin_operator
scoop spoon arm bowl
1
0 0
1
0 3 -1 0
1
end_operator
0
