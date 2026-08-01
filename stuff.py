# google doc parsing library bs4 was researched for this question. 

import requests
from bs4 import BeautifulSoup

def get_dimensions(table: list[list[str]]) -> tuple[int, int]:
  # Gets the maximum y and x value for the table
  x_vals = []
  y_vals = []
  for row in table[1:]:
    x_vals.append(int(row[0]))
    y_vals.append(int(row[2]))
  return max(x_vals) + 1, max(y_vals) + 1
    
def get_empty_print_table(table: list[list[str]], dims: tuple[int, int]) -> list[list[str]]:
  # Gets a list in the shape of our
  return [[" " for _ in range(dims[0])] for _ in range(dims[1])]
  

def print_doc_decoder(doc_url: str):
  response = requests.get(doc_url)
  parser = BeautifulSoup(response.text, "html.parser")

  table_data = []
  table = parser.find("table")

  # get data
  if table:
    for row in table.find_all("tr"):
      row_data = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
      table_data.append(row_data)

  # parse data into table resembling output
  (x_max, y_max) = get_dimensions(table_data)
  print(x_max, y_max)
  print_table = get_empty_print_table(table_data, (x_max, y_max))
  for row in table_data[1:]:
      x = int(row[0])
      y = int(row[2])
      print_table[(y_max - 1) - y][x] = row[1]

  # final print
  print(print_table)
  output = ""
  for row in print_table:
    for val in row:
      output += val
    output += "\n"
  print(output)

print_doc_decoder("https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub")