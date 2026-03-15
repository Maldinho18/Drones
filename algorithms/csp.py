from __future__ import annotations
from collections import deque

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    # TODO: Implement your code here
    
    asignado: dict[str, str] = {}
    def backtrack() -> dict[str, str] | None:
      if csp.is_complete(asignado):
        return dict(asignado)
      
      noAsignado = csp.get_unassigned_variables(asignado)
      if not noAsignado:
        return dict(asignado)
      
      var = noAsignado[0]
      
      for valor in csp.domains[var]:
        if csp.is_consistent(var, valor, asignado):
          csp.assign(var, valor, asignado)
          resultado = backtrack()
          if resultado is not None:
            return resultado
          csp.unassign(var, asignado)
      return None
    return backtrack()


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    # TODO: Implement your code here
    asignacion: dict[str, str] = {}
    def forward_check(var: str) -> bool:
      for vecino in csp.get_neighbors(var):
        if vecino in asignacion:
          continue
        
        nuevo_dominio: list[str] = []
        for val in csp.domains[vecino]:
          if csp.is_consistent(vecino, val, asignacion):
            nuevo_dominio.append(val)
            
        csp.domains[vecino] = nuevo_dominio
        if len(csp.domains[vecino]) == 0:
          return False
      return True
    
    def backtrack() -> dict[str, str] | None:
      if csp.is_complete(asignacion):
        return dict(asignacion)
      
      no_asignado = csp.get_unassigned_variables(asignacion)
      if not no_asignado:
        return dict(asignacion)
      
      var = no_asignado[0]
      
      for valor in list(csp.domains[var]):
        if csp.is_consistent(var, valor, asignacion):
          csp.assign(var, valor, asignacion)
          
          dominios_guardados = {v: list(csp.domains[v]) for v in csp.variables}
          
          ok = forward_check(var)
          if ok:
            resultado = backtrack()
            if resultado is not None:
              return resultado
          
          csp.domains = dominios_guardados
          csp.unassign(var, asignacion)
      return None
    
    return backtrack()
    


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
   
    assignment: dict[str, str] = {}

    def revise(xi: str, xj: str, queue_assignment: dict[str, str]) -> bool:
       
        revised = False
        for x in list(csp.domains[xi]):
            supported = False
            temp = dict(queue_assignment)
            temp[xi] = x
            for y in csp.domains[xj]:
                # ¿puedo asignar xj=y dado que xi=x ya está puesto?
                if csp.is_consistent(xj, y, temp):
                    supported = True
                    break
            if not supported:
                csp.domains[xi].remove(x)
                revised = True
        return revised

    def ac3(initial_queue=None, queue_assignment=None) -> bool:
        if queue_assignment is None:
            queue_assignment = {}

        if initial_queue is None:
            q = deque()
            for xi in csp.variables:
                for xj in csp.get_neighbors(xi):
                    q.append((xi, xj))
        else:
            q = deque(initial_queue)

        while q:
            xi, xj = q.popleft()
            if revise(xi, xj, queue_assignment):
                if len(csp.domains[xi]) == 0:
                    return False
                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        q.append((xk, xi))
        return True

    # AC-3 global antes de iniciar
    saved_domains0 = {v: list(csp.domains[v]) for v in csp.variables}
    if not ac3(queue_assignment=assignment):
        csp.domains = saved_domains0
        return None

    def backtrack() -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)

        unassigned = csp.get_unassigned_variables(assignment)
        if not unassigned:
            return dict(assignment)

        var = unassigned[0]

        for value in list(csp.domains[var]):
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)

                saved_domains = {v: list(csp.domains[v]) for v in csp.variables}

                # AC3 local: arcos (neighbor, var) para vecinos no asignados
                local_queue = []
                for neighbor in csp.get_neighbors(var):
                    if neighbor not in assignment:
                        local_queue.append((neighbor, var))

                ok = ac3(initial_queue=local_queue, queue_assignment=assignment)
                if ok:
                    result = backtrack()
                    if result is not None:
                        return result

                csp.domains = saved_domains
                csp.unassign(var, assignment)

        return None

    return backtrack()
    

def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:

    assignment: dict[str, str] = {}

    def select_mrv_variable() -> str:

        unassigned = csp.get_unassigned_variables(assignment)

        best_var = unassigned[0]
        best_mrv = float("inf")
        best_degree = -1

        for var in unassigned:
            legal_vals = 0
            for val in csp.domains[var]:
                if csp.is_consistent(var, val, assignment):
                    legal_vals += 1

            degree = sum(1 for nb in csp.get_neighbors(var) if nb not in assignment)

            if legal_vals < best_mrv:
                best_mrv = legal_vals
                best_var = var
                best_degree = degree
            elif legal_vals == best_mrv and degree > best_degree:
                best_var = var
                best_degree = degree

        return best_var

    def order_lcv_values(var: str) -> list[str]:
      
        return sorted(list(csp.domains[var]), key=lambda v: csp.get_num_conflicts(var, v, assignment))

    def forward_check(var: str) -> bool:
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            new_domain: list[str] = []
            for val in csp.domains[neighbor]:
                if csp.is_consistent(neighbor, val, assignment):
                    new_domain.append(val)

            csp.domains[neighbor] = new_domain
            if len(csp.domains[neighbor]) == 0:
                return False
        return True

    def backtrack() -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return dict(assignment)

        var = select_mrv_variable()

        for value in order_lcv_values(var):
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)

                saved_domains = {v: list(csp.domains[v]) for v in csp.variables}

                ok = forward_check(var)
                if ok:
                    result = backtrack()
                    if result is not None:
                        return result

                csp.domains = saved_domains
                csp.unassign(var, assignment)

        return None

    return backtrack()
    
