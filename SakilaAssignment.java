package mysqlExample;

import java.sql.Connection;
import java.sql.Statement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.DriverManager;
import java.sql.PreparedStatement;

public class SakilaAssignment {

	// attributes
	private Connection connect = null;
	private ResultSet resultSet = null;
	//private int resultInsert; // Adding variable for displaying insert query result
	private Statement statement = null;
	
	public void readDatabase() throws Exception{
		
		try {
			// This will load the MySQL driver, each DB has its own driver
			Class.forName("com.mysql.cj.jdbc.Driver");
			
			// setup connection to sakila database
			//connect = DriverManager.getConnection("jdbc:mysql://localhost/sakila?user=sakilauser&password=sakila123");
			connect = DriverManager.getConnection("jdbc:mysql://localhost/sakila","sakilauser", "sakila123");
			// Statement to issue SQL queries to the database
			statement = connect.createStatement();
			
			// Result set gets the result of the SQL query
			resultSet = statement.executeQuery("select * from sakila.staff");
			//resultInsert = statement.executeUpdate("Insert into sakila.staff (staff_id,first_name,last_name, address_id, email,store_id, active,username,password) VALUES(3, 'Sonam', 'Gupta', 3, 'SGupta@my.harrisburgu.edu', 1,1,'ssg1105', 'coffee123' )");
			
			// output results
			writeResultSet(resultSet);
			
		} catch(Exception e) {
			throw e;
		} finally {
			close();
		}
		
	}
	
	// Function to insert record in the database
	private PreparedStatement preparedStatement = null;
	
	public void insertRecord() throws Exception {

	    try {
			// This will load the MySQL driver, each DB has its own driver
			Class.forName("com.mysql.cj.jdbc.Driver");
			
			// setup connection to sakila database
			
			connect = DriverManager.getConnection("jdbc:mysql://localhost/sakila","sakilauser", "sakila123");
			// Statement to issue SQL queries to the database
			Statement st = connect.createStatement();
			//preparedStatement = connect.prepareStatement("Insert into sakila.staff (staff_id,first_name,last_name, address_id, email,store_id, active,username,password) VALUES(3, 'Sonam', 'Gupta', 3, 'SGupta@my.harrisburgu.edu', 1,1,'ssg1105', 'coffee123' )");
			
			st.executeUpdate("Insert into sakila.staff (staff_id,first_name,last_name, address_id, email,store_id, active,username,password) VALUES(3, 'Sonam', 'Gupta', 3, 'SGupta@my.harrisburgu.edu', 1,1,'ssg1105', 'coffee123' )");
			connect.commit();
			
			// output results
			resultSet = statement.executeQuery("select * from sakila.staff");
			writeResultSet(resultSet);

	    } finally {
	    	close();
	    }
	}
	 
	
	private void writeResultSet(ResultSet rs) throws SQLException{
		// ResultSet cursor is initially located before the first data row
		while(resultSet.next()) {
			// Its possible to access data via column name
			// or via column no. Numbers start at 1
			String staff_id = resultSet.getString("staff_id");
			String first = resultSet.getString("first_name");
			String last = resultSet.getString("last_name");
			int address_id = resultSet.getInt("address_id");
			//Blob picture = resultSet.getBl;
			String email = resultSet.getString("email");
			int active = resultSet.getInt("active");
			String username = resultSet.getString("username");
			String password = resultSet.getString("password");
			String last_update = resultSet.getString("last_update");
			
			
			// write out actual variables
			System.out.println("Staff ID: " + staff_id);
			System.out.println("First name: " + first);
			System.out.println("Last name: " + last);
			System.out.println("Address ID: " + address_id);
			System.out.println("Email: " + email);
			System.out.println("Active: " + active);
			System.out.println("Username: " + username);
			System.out.println("Password: " + password);
			System.out.println("Last Update: " + last_update + "\n");
			
		}
	}
	
	// Close all resources
	private void close() {
		try {
			if (resultSet != null) {
				resultSet.close();
			}
			if(statement != null) {
				statement.close();				
			}
			if(connect != null) {
				connect.close();
			}
		} catch(Exception e) {
			
		}
	}
	
	public static void main(String[] args) throws Exception{
		// create an MySQL access object
		SakilaAssignment dbAccess = new SakilaAssignment();
		dbAccess.readDatabase();

	}

}