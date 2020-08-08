(define (domain handy)
    (:predicates (GRASPABLE ?x)
                 (carry ?x ?y)
		 (free ?x)
                 (SCOOPABLE ?x)
                 (SUPPORTABLE ?x)
                 (CONTAINABLE ?x)
		 (contains ?x ?y)
                 (scooped ?x ?y))

    (:action pickup :parameters(?x ?y)
        :precondition (and (GRASPABLE ?x)
	                   (free ?y))
			   
        :effect       (and (carry ?y ?x)
	                   (not (free ?y))
	                   (not (GRASPABLE ?x))))

    (:action dropoff :parameters (?x ?y ?z)
        :precondition (and (carry ?y ?x)
	                   (CONTAINABLE ?z))
			   
	:effect       (and (contains ?z ?x)
                           (GRASPABLE ?x)
	                   (free ?y)
			   (not (carry ?y ?x))))

    (:action scoop :parameters (?x ?y ?z)
        :precondition (and (carry ?y ?x)
	                   (SUPPORTABLE ?z))
			   
	:effect       (and (scooped ?z ?x)))
)
	                
