(define (problem handy_vision)
    (:domain handy)
    (:objects arm trowel spoon bowl )
    (:init (free arm) (GRASPABLE trowel) (SCOOPABLE trowel) (SUPPORTABLE trowel) (GRASPABLE spoon) (SCOOPABLE spoon) (CONTAINABLE spoon) (SUPPORTABLE bowl) )
    (:goal (and (scooped bowl spoon))))