$(document).ready(function() {
  $('table').DataTable({
    pageLength: 10,
    lengthMenu: [5, 10, 20, 50],
    order: [[0, 'desc']],
    dom: 'Bfrtip',
    buttons: [
      'copy', 'csv', 'excel', 'pdf', 'print'
    ]
  });
});
